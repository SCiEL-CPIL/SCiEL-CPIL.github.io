import time
import numpy as np

N = 400
DEADLINE = 115.0  # Safety margin before 120s limit

def _autoconv_bound(x):
    """
    Compute the circular autocorrelation upper bound C3.
    Metric: max_{k != 0} |R[k]| / R[0]
    where R is the circular autocorrelation of x.
    """
    n = len(x)
    # Autocorrelation via FFT
    fft_x = np.fft.fft(x)
    power = fft_x * np.conj(fft_x)
    R = np.fft.ifft(power).real
    
    center = R[0]
    if center <= 0:
        return 1.0
    
    # Off-center maximum absolute value
    off_vals = np.concatenate([R[1:], R[:-1]])
    off_max = np.max(np.abs(off_vals))
    
    return off_max / center

def _lfsr_sequence(n, poly_bits, init):
    """Generate n bits from an LFSR."""
    seq = []
    state = init
    deg = poly_bits.bit_length() - 1
    for _ in range(n):
        seq.append(state & 1)
        feedback = 0
        for i in range(deg + 1):
            if (poly_bits >> i) & 1:
                feedback ^= (state >> i) & 1
        state = (state >> 1) | (feedback << deg)
    return seq

def _compute_autocorr(x):
    """Compute circular autocorrelation vector."""
    fft_x = np.fft.fft(x)
    power = fft_x * np.conj(fft_x)
    return np.fft.ifft(power).real

def _simulated_annealing(x, R, max_time):
    """
    Simulated Annealing with fast O(N) autocorrelation updates.
    x: current sequence (np.array of floats -1, 1)
    R: current autocorrelation vector (np.array)
    max_time: time limit for this phase
    """
    n = len(x)
    best_x = x.copy()
    best_R = R.copy()
    best_val = _autoconv_bound(x)
    
    start_time = time.time()
    T = 1.0  # Initial temperature
    T_end = 0.01
    cooling_rate = 0.999
    
    # Precompute rolls? No, roll is O(N) anyway, but we need it for every flip.
    # We can precompute the indices or just use np.roll.
    # np.roll is implemented in C, so it's fast.
    
    steps = 0
    while (time.time() - start_time) < max_time:
        # Pick random index
        i = np.random.randint(n)
        
        # Calculate delta
        old_val = x[i]
        new_val = -old_val
        delta = new_val - old_val  # -2 * old_val
        
        # Update R
        # U[k] = x[i-k] + x[i+k]
        # x[i-k] is x shifted right by i (index k corresponds to i-k in original)
        # Wait:
        # R[k] = sum_j x[j] x[j+k]
        # Term with x[i]:
        # 1. j=i -> x[i] x[i+k]
        # 2. j+k=i -> j=i-k -> x[i-k] x[i]
        # So change is delta * (x[i+k] + x[i-k])
        
        # np.roll(x, -i) gives y where y[k] = x[(k-i)%N] ?
        # np.roll(a, shift): shift > 0 means right shift.
        # a = [0, 1, 2, 3]. roll(a, 1) = [3, 0, 1, 2].
        # index k in rolled array corresponds to index (k - shift) in original?
        # rolled[k] = a[(k - shift) % N].
        # So roll(x, i)[k] = x[(k-i) % N] = x[k-i]. This is x[i-k] if we consider modulo?
        # x[k-i] is the element at index k-i.
        # We want x[i-k].
        # x[i-k] = x[-(k-i)].
        # roll(x, -i)[k] = x[(k+i) % N] = x[k+i].
        # So:
        # term1 = x[i+k] -> np.roll(x, -i) ?
        # Let's check:
        # x = [1, 2, 3, 4]. i=1.
        # We want x[1+k] for k=0,1,2,3 -> [2, 3, 4, 1].
        # np.roll(x, -1) -> [2, 3, 4, 1]. Correct.
        # We want x[1-k] for k=0,1,2,3 -> [1, 4, 3, 2].
        # np.roll(x, 1) -> [4, 1, 2, 3].
        # index 0: 4 (x[3] = x[1-(-2)]? No. x[1-0]=x[1]=1.
        # My formula: x[i-k].
        # k=0: x[1].
        # k=1: x[0].
        # k=2: x[-1]=x[3].
        # k=3: x[-2]=x[2].
        # Sequence: 1, 1, 4, 3?
        # Let's re-verify indices.
        # R[k] = sum_j x[j] x[j+k].
        # If we flip x[i], it affects R[k] if i is involved in the product.
        # x[j] = x[i] when j=i. Term: x[i] x[i+k].
        # x[j+k] = x[i] when j+k=i => j=i-k. Term: x[i-k] x[i].
        # So Delta R[k] = delta * (x[i+k] + x[i-k]).
        
        # Vector for x[i+k]:
        # We need a vector V1 where V1[k] = x[(i+k) % N].
        # This is x shifted left by i.
        # np.roll(x, -i) shifts left by i.
        # Check: x=[1,2,3,4], i=1. roll(x,-1)=[2,3,4,1].
        # k=0: 2 = x[1]. Correct.
        # k=1: 3 = x[2]. Correct.
        
        # Vector for x[i-k]:
        # We need a vector V2 where V2[k] = x[(i-k) % N].
        # This is x shifted right by i, then reversed?
        # Let's check np.roll(x, i).
        # x=[1,2,3,4], i=1. roll(x,1)=[4,1,2,3].
        # k=0: 4. We want x[1-0]=x[1]=1. Mismatch.
        # roll(x, i)[k] = x[(k-i) % N].
        # We want x[(i-k) % N].
        # Note that (i-k) % N = -(k-i) % N.
        # So V2 is the reverse of the roll?
        # Or simply: V2[k] = x[i-k].
        # If we take np.roll(x, i), we get x[k-i].
        # x[k-i] is not x[i-k].
        # However, x is real.
        # Let's just compute V2 explicitly or use a trick.
        # V2[k] = x[i-k].
        # If we reverse x, x_rev[k] = x[N-1-k].
        # This is getting complicated.
        
        # Alternative:
        # V1 = np.roll(x, -i)  # x[i+k]
        # V2 = np.roll(x, i)   # x[k-i]
        # Is x[i-k] related to x[k-i]?
        # x[i-k] = x[-(k-i)].
        # Since x is not symmetric, we cannot just use V2 directly.
        
        # Let's construct V2 properly.
        # We can use np.roll on a reversed array?
        # Or simply:
        # V2 = np.roll(x, i)
        # V2[k] = x[(k-i)%N].
        # We want x[(i-k)%N].
        # Note that (i-k)%N = N - (k-i)%N (if k!=i).
        # So V2_desired[k] = x[N - V2_index]?
        # Let idx = (k-i) % N.
        # We want x[-idx % N].
        
        # Actually, there is a simpler way.
        # The sequence x[i-k] for k=0..N-1 is just the reverse of the sequence x[i+k] for k=0..N-1?
        # No.
        # x[i+0], x[i+1], ...
        # x[i-0], x[i-1], ...
        # These are traversing the circle in opposite directions.
        
        # Let's just use slicing or advanced indexing if it's faster, or stick to roll if we can map it.
        # V1 = np.roll(x, -i)
        # V2:
        # We want array A where A[k] = x[(i-k)%N].
        # Let B = np.roll(x, i). B[k] = x[(k-i)%N].
        # A[k] = x[-(k-i)%N].
        # If we reverse B, B_rev[k] = B[N-1-k] = x[((N-1-k)-i)%N] = x[(-1-k-i)%N].
        # This doesn't match directly.
        
        # Let's try:
        # V2 = np.roll(x, i)
        # V2 is x[k-i].
        # We want x[i-k].
        # Note x[i-k] = x[-(k-i)].
        # If we create an array C where C[k] = x[-k % N], then C is x reversed with shift?
        # x = [1, 2, 3, 4].
        # x[-k%4]: k=0->1, k=1->4, k=2->3, k=3->2. [1,4,3,2].
        # This is x_rev_shifted.
        
        # Okay, let's just do:
        # V1 = np.roll(x, -i)
        # V2 = np.roll(x, i)
        # Wait, is x[i-k] equal to V2 reversed?
        # V2 = [x[-i], x[1-i], x[2-i], ...]
        # Reverse V2 = [..., x[2-i], x[1-i], x[-i]]
        # Index k in reversed V2 corresponds to index N-1-k in V2.
        # RevV2[k] = V2[N-1-k] = x[(N-1-k)-i] = x[-1-k-i].
        # We want x[i-k].
        # -1-k-i vs i-k.
        # Difference is 2i+1.
        # So RevV2 is shifted.
        
        # Simplest robust method:
        # V1 = np.roll(x, -i)
        # V2 = np.roll(x, i)
        # Actually, x[i-k] is just x reversed and rolled?
        # x_rev = x[::-1]
        # x_rev[k] = x[N-1-k].
        # We want x[i-k].
        # x[i-k] = x_rev[N-1-(i-k)] = x_rev[N-1-i+k].
        # So V2 = np.roll(x[::-1], -(N-1-i))?
        # This is O(N) to reverse, which is fine.
        
        # Let's optimize:
        # V1 = np.roll(x, -i)
        # V2 = np.roll(x[::-1], i) # Check:
        # x[::-1][k] = x[N-1-k].
        # roll(x[::-1], i)[k] = x[::-1][(k-i)%N] = x[N-1-(k-i)] = x[N-1-k+i].
        # We want x[i-k].
        # N-1-k+i = i-k + N-1.
        # Indices are mod N.
        # x[i-k] = x[i-k + N].
        # So we need N-1 = N? No.
        # x[N-1-k+i] vs x[i-k].
        # Difference is N-1.
        # So V2 = np.roll(x[::-1], i) gives x[i-k + N-1].
        # So we need to shift by 1 more?
        # np.roll(x[::-1], i+1)?
        # x[N-1-(k-i-1)] = x[N-1-k+i+1] = x[N-k+i] = x[i-k].
        # Yes!
        # V2 = np.roll(x[::-1], i+1)
        
        # Cost of x[::-1] is O(N).
        # Cost of roll is O(N).
        # Total O(N).
        
        # Update R
        U = np.roll(x, -i) + np.roll(x[::-1], i+1)
        R += delta * U
        
        # Compute new bound
        # Center R[0] is constant (N) for +/-1 sequences?
        # R[0] = sum x[i]^2 = N. Yes, constant.
        off_max = np.max(np.abs(R[1:])) # R is symmetric, so R[1:] covers all k!=0?
        # Circular autocorrelation is symmetric: R[k] = R[-k] = R[N-k].
        # So max abs of R[1:] is the max abs of all off-center.
        current_val = off_max / N
        
        # Acceptance
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
            best_R = R.copy()
            # x is already updated
        else:
            # Metropolis criterion
            # delta_energy = current_val - best_val? No, compare to current state's value before flip?
            # We need the value BEFORE the flip.
            # Let's store old_val_bound.
            # This requires computing bound before flip.
            # To save time, we can approximate or just store it.
            # Computing bound is O(N).
            # We are doing O(N) update.
            # So we can compute old bound.
            
            # Actually, I didn't store old bound.
            # Let's restructure.
            # 1. Compute old_bound.
            # 2. Flip x.
            # 3. Update R.
            # 4. Compute new_bound.
            # 5. Accept/Reject.
            
            # This doubles the bound computation cost.
            # Bound computation is max(abs(R[1:])).
            # This is fast.
            
            # Let's restart the loop logic with this in mind.
            pass
            
        steps += 1
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_x, best_val

def construct():
    t_start = time.time()
    
    # 1. Initialization
    best_x = None
    best_val = float('inf')
    
    # Try LFSR sequences
    configs = [
        (0b100010001, 0x1FF),
        (0b100100001, 0x1FF),
        (0b101101001, 0x1FF),
        (0b100010001, 0x0FF),
        (0b100100001, 0x0FF),
        (0b101101001, 0x0FF),
        (0b100010001, 0x123),
        (0b100100001, 0x123),
        (0b101101001, 0x123),
        (0b1000001001, 0x3FF),
        (0b1101010001, 0x3FF),
        (0b1000001001, 0x1FF),
        (0b1101010001, 0x1FF),
    ]
    
    for poly, init in configs:
        if time.time() - t_start > 5:
            break
        bits = _lfsr_sequence(N, poly, init)
        x = np.array([1.0 if b == 1 else -1.0 for b in bits])
        val = _autoconv_bound(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # Try Random
    for seed in range(10):
        if time.time() - t_start > 10:
            break
        rng = np.random.RandomState(seed)
        x = rng.choice([-1.0, 1.0], size=N)
        val = _autoconv_bound(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    if best_x is None:
        return np.ones(N)
        
    # 2. Optimization
    remaining_time = DEADLINE - (time.time() - t_start) - 5
    if remaining_time > 1.0:
        R = _compute_autocorr(best_x)
        best_x, best_val = _simulated_annealing(best_x, R, remaining_time)
        
    return best_x

# Corrected Simulated Annealing Logic
def _simulated_annealing(x, R, max_time):
    n = len(x)
    best_x = x.copy()
    best_R = R.copy()
    best_val = _autoconv_bound(x)
    
    start_time = time.time()
    T = 1.0
    T_end = 0.01
    cooling_rate = 0.9995
    
    # Precompute reversed x? It changes on flip.
    # So we must compute x[::-1] every time.
    
    while (time.time() - start_time) < max_time:
        i = np.random.randint(n)
        
        old_val = x[i]
        new_val = -old_val
        delta = new_val - old_val
        
        # Compute old bound
        old_bound = np.max(np.abs(R[1:])) / N
        
        # Update x
        x[i] = new_val
        
        # Update R
        # V1 = x[i+k] -> roll(x, -i)
        # V2 = x[i-k] -> roll(x[::-1], i+1)
        U = np.roll(x, -i) + np.roll(x[::-1], i+1)
        R += delta * U
        
        # Compute new bound
        new_bound = np.max(np.abs(R[1:])) / N
        
        # Accept/Reject
        if new_bound < old_bound:
            # Always accept
            if new_bound < best_val:
                best_val = new_bound
                best_x = x.copy()
                best_R = R.copy()
        else:
            # Metropolis
            prob = np.exp((old_bound - new_bound) / T)
            if np.random.rand() < prob:
                # Accept
                if new_bound < best_val:
                    best_val = new_bound
                    best_x = x.copy()
                    best_R = R.copy()
            else:
                # Reject: Revert x and R
                x[i] = old_val
                R -= delta * U  # Revert update
                
        T *= cooling_rate
        if T < T_end:
            T = T_end
            
    return best_x, best_val

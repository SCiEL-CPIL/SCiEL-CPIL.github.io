"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 256
K = 4

# Precompute all 4-term arithmetic progressions (a, a+d, a+2d, a+3d) within 1..256
# We store them as 0-indexed tuples (i, i+d, i+2d, i+3d)
APs = []
ap_indices_by_pos = [[] for _ in range(N)]

for d in range(1, N):
    max_a = N - 3 * d
    if max_a < 1:
        break
    for a in range(1, max_a + 1):
        i1 = a - 1
        i2 = i1 + d
        i3 = i2 + d
        i4 = i3 + d
        ap_id = len(APs)
        APs.append((i1, i2, i3, i4))
        ap_indices_by_pos[i1].append(ap_id)
        ap_indices_by_pos[i2].append(ap_id)
        ap_indices_by_pos[i3].append(ap_id)
        ap_indices_by_pos[i4].append(ap_id)

TOTAL_APS = len(APs)

def count_mono_4ap_fast(coloring):
    """Count monochromatic 4-term APs in a 0/1 coloring of 1..256."""
    c = np.array(coloring, dtype=np.int8)
    count = 0
    for d in range(1, N):
        max_a = N - 3 * d
        if max_a < 1:
            break
        seq1 = c[:max_a]
        seq2 = c[d:d+max_a]
        seq3 = c[2*d:2*d+max_a]
        seq4 = c[3*d:3*d+max_a]
        mono = (seq1 == seq2) & (seq2 == seq3) & (seq3 == seq4)
        count += np.sum(mono)
    return count

def discrete_to_continuous(x):
    """Map binary vector to [-1, 1] continuous space."""
    return 2.0 * x - 1.0

def continuous_to_discrete(x):
    """Map continuous vector to binary vector (threshold at 0)."""
    return (x >= 0).astype(np.int8)

def evaluate_continuous(x):
    """
    Evaluate the QUBO objective in continuous space.
    Objective: Sum over all APs of (x1 + x2 + x3 + x4)^2 / 16 
    Wait, the monochromatic condition in {0,1} is x1==x2==x3==x4.
    In {-1, 1} space, monochromatic means all +1 or all -1.
    Let's use the indicator that all are equal.
    If we use (x1+x2+x3+x4)^2, it is 16 if all same, 0 if mixed (2+2), 4 if (3+1).
    This penalizes mixed configurations.
    Specifically, for 4 terms:
    All same: sum = +/-4, sq = 16.
    3 same, 1 diff: sum = +/-2, sq = 4.
    2 same, 2 diff: sum = 0, sq = 0.
    
    We want to MINIMIZE monochromatic APs.
    So we want to maximize the "mixedness".
    Let's define the cost function as the number of monochromatic APs.
    Indicator for mono in {0,1}: I(x1==x2==x3==x4).
    In {-1,1}, let y_i = x_i.
    Mono iff y1*y2*y3*y4 == 1 AND (y1==y2 OR y1==-y2 is false... no).
    
    Let's stick to the {0,1} formulation but relax it.
    Let z_i in [0,1] be the probability of color 1.
    The expected number of mono APs is not simply additive due to dependencies.
    However, a common relaxation for "minimize number of satisfied clauses" where clause is "all equal" is to use the product of differences.
    
    Actually, let's look at the gradient of the discrete objective.
    Let E(x) = sum_AP I(x1==x2==x3==x4).
    This is step-like.
    
    Alternative: Use the "spin glass" formulation.
    Let s_i in {-1, 1}.
    An AP is monochromatic if s1=s2=s3=s4.
    This happens if s1*s2 = 1, s2*s3 = 1, s3*s4 = 1.
    Or s1*s2 = -1, s2*s3 = -1, s3*s4 = -1.
    
    Let's define a penalty for NOT being monochromatic? No, we want to minimize mono.
    So we want to MAXIMIZE the number of non-mono APs.
    An AP is non-mono if at least one adjacent pair is different.
    
    Let's try a different continuous proxy.
    Consider the function f(x) = sum_AP (1 - (x1*x2*x3*x4)) ? No.
    
    Let's use the standard QUBO mapping for "all equal".
    In {0,1}, x1==x2==x3==x4 is equivalent to:
    x1 x2 + x2 x3 + x3 x4 + x1 x4 + x2 x4 + x1 x3 - 2(x1+x2+x3+x4) + 4 ? No.
    
    Let's use the identity:
    I(x1==x2==x3==x4) = (1/4) * ( (x1+x2+x3+x4)^2 - 2 * sum_{i<j} (xi-xj)^2 ? No.
    
    Let's go back to the discrete gradient approach but in a continuous "annealing" style.
    We will maintain a vector v of real numbers in [0,1].
    We define the energy E(v) = sum_AP P_mono(v_AP).
    Where P_mono is the probability that 4 independent Bernoulli(v_i) variables are equal.
    P(all 0) = prod(1-v_i)
    P(all 1) = prod(v_i)
    P_mono = prod(1-v_i) + prod(v_i).
    
    This is a smooth, differentiable function!
    Gradient with respect to v_k for an AP (i,j,l,m) containing k:
    d/dv_k [ prod(1-v) + prod(v) ]
    = -prod_{m!=k}(1-v_m) * (1-v_k)^{-1} * prod(1-v)  ? No, product rule.
    Let A = prod(1-v), B = prod(v).
    d/dv_k (A+B) = d/dv_k A + d/dv_k B
    d/dv_k A = -A / (1-v_k)
    d/dv_k B = B / v_k
    
    So grad_k = sum_{APs containing k} ( -A_AP / (1-v_k) + B_AP / v_k )
    
    We want to MINIMIZE E(v).
    We can use Gradient Descent with momentum.
    To ensure we stay in [0,1], we can use a sigmoid or clip.
    Clipping is simpler and often works well for QUBO relaxations if we occasionally snap to discrete.
    
    Strategy:
    1. Initialize v randomly in [0.2, 0.8].
    2. Iteratively compute gradient and update v = v - lr * grad.
    3. Clip v to [0.1, 0.9] to avoid division by zero.
    4. Periodically snap to discrete, evaluate, and if better, restart continuous from there?
    Or just run continuous for a long time and snap at the end.
    The continuous landscape may have basins that correspond to good discrete solutions.
    
    Let's implement this.
    """
    pass

def subgradient_search(deadline):
    # Precompute AP indices for fast vectorization?
    # We have APs as list of tuples.
    # We can represent v as a numpy array of shape (N,).
    # For each step, we need to compute the sum of gradients for each position.
    
    # To speed up gradient computation, we can precompute for each position 'pos',
    # the list of APs it belongs to, and for each AP, the other 3 indices.
    # But the gradient depends on the current values v.
    
    # Let's try a vectorized approach.
    # v is (N,).
    # We need to compute for each AP: A = prod(1-v_AP), B = prod(v_AP).
    # Then grad_pos = sum_{AP in pos} ( -A/(1-v_pos) + B/v_pos ).
    
    # This is O(TOTAL_APS * 4) per step.
    # TOTAL_APS is roughly N^2/3 ~ 20000.
    # 20000 * 4 = 80000 ops per step.
    # In Python, a loop of 20000 is slow. We need numpy vectorization.
    
    # Let's create a matrix M of shape (TOTAL_APS, 4) containing the indices.
    M = np.array(APs, dtype=np.int32)
    
    def compute_gradient(v):
        # v: (N,)
        # v_AP: (TOTAL_APS, 4)
        v_ap = v[M]
        
        # Compute products
        # To avoid underflow/overflow in products of 4 terms, it's fine.
        ones = 1.0 - v_ap
        A = np.prod(ones, axis=1)
        B = np.prod(v_ap, axis=1)
        
        # We need to sum contributions to each position.
        # grad[i] = sum_{ap where i in ap} ( -A[ap]/(1-v[i]) + B[ap]/v[i] )
        
        # Let's compute the term for each AP and each of its 4 positions.
        # term_k = -A/(1-v_k) + B/v_k
        
        # We can compute a matrix G of shape (TOTAL_APS, 4)
        # G[:, j] = -A[:, None] / (1 - v_ap[:, j]) + B[:, None] / v_ap[:, j]
        
        # Clip v_ap to avoid division by zero
        v_safe = np.clip(v_ap, 1e-6, 1 - 1e-6)
        
        G = -A[:, None] / (1.0 - v_safe) + B[:, None] / v_safe
        
        # Now sum over APs for each position.
        # We can use np.bincount
        # We need to sum G[:, 0] for indices in M[:, 0], etc.
        
        grad = np.zeros(N, dtype=np.float64)
        
        # Unroll the 4 positions for speed
        # Note: bincount weights must be 1D.
        # We can do 4 bincounts.
        
        # However, bincount with weights is fast in numpy.
        
        # pos 0
        np.add.at(grad, M[:, 0], G[:, 0])
        # pos 1
        np.add.at(grad, M[:, 1], G[:, 1])
        # pos 2
        np.add.at(grad, M[:, 2], G[:, 2])
        # pos 3
        np.add.at(grad, M[:, 3], G[:, 3])
        
        return grad

    def compute_energy(v):
        v_ap = v[M]
        ones = 1.0 - v_ap
        A = np.prod(ones, axis=1)
        B = np.prod(v_ap, axis=1)
        return np.sum(A + B)

    # Main Loop
    # Multiple restarts to escape local minima
    best_discrete_score = float('inf')
    best_discrete_coloring = None
    
    # Number of restarts depends on time
    # Each restart: continuous descent for T steps.
    
    # Estimate time per step
    # Let's do a quick timing
    v_test = np.random.rand(N)
    start = time.time()
    for _ in range(10):
        _ = compute_gradient(v_test)
    end = time.time()
    time_per_step = (end - start) / 10.0
    
    # Target steps per restart: enough to converge somewhat, but allow many restarts
    # If time_per_step is 0.001s, we can do 1000 steps in 1s.
    # Let's aim for 500-1000 steps per restart.
    
    max_steps_per_restart = 800
    
    # Learning rate schedule: start high, decay? Or constant with momentum?
    # Let's use Adam-like optimizer or simple momentum.
    # Simple momentum: v = v - lr * grad + momentum * v_prev_delta
    
    # We'll use a fixed learning rate that decays slightly.
    
    for restart in range(100): # Max restarts, will break on time
        if time.time() > deadline - 10:
            break
            
        # Initialize v
        # Random start
        v = np.random.uniform(0.2, 0.8, N)
        
        # Momentum
        velocity = np.zeros(N, dtype=np.float64)
        momentum = 0.9
        lr = 0.05
        
        # Run continuous descent
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            grad = compute_gradient(v)
            
            # Update
            velocity = momentum * velocity - lr * grad
            v = v + velocity
            
            # Clip to [0.01, 0.99] to keep probabilities valid
            v = np.clip(v, 0.01, 0.99)
            
            # Decay learning rate
            lr *= 0.999
            
        # End of restart
        # Snap to discrete
        discrete_v = (v >= 0.5).astype(np.int8)
        
        # Evaluate discrete score
        score = count_mono_4ap_fast(discrete_v)
        
        if score < best_discrete_score:
            best_discrete_score = score
            best_discrete_coloring = discrete_v.copy()
            
        # Optional: Local search on the best discrete solution found so far?
        # The continuous method might find a good basin.
        # Let's add a small local search polish at the end of each restart if time permits.
        # But the prompt asks for a different strategy.
        # The continuous method itself is the new strategy.
        # However, combining it with a final global local search is standard practice.
        # Let's stick to pure continuous + snap for now to test the hypothesis.
        
    # Final Polish: Run a standard local search on the best found so far
    if best_discrete_coloring is not None:
        # Reuse the local search logic from the seed, but simplified
        # We'll implement a simple bit-flip local search here
        c = best_discrete_coloring.copy()
        improved = True
        steps = 0
        current_score = best_discrete_score
        
        while improved and steps < 2000 and time.time() < deadline - 2:
            improved = False
            best_pos = -1
            best_delta = 0
            
            for pos in range(N):
                # Calculate delta for flipping pos
                old_bit = c[pos]
                target_bit = 1 - old_bit
                delta = 0
                
                for ap_id in ap_indices_by_pos[pos]:
                    i1, i2, i3, i4 = APs[ap_id]
                    
                    if i1 == pos:
                        b1, b2, b3 = c[i2], c[i3], c[i4]
                    elif i2 == pos:
                        b1, b2, b3 = c[i1], c[i3], c[i4]
                    elif i3 == pos:
                        b1, b2, b3 = c[i1], c[i2], c[i4]
                    else:
                        b1, b2, b3 = c[i1], c[i2], c[i3]
                        
                    if b1 == old_bit and b2 == old_bit and b3 == old_bit:
                        delta -= 1
                    elif b1 == target_bit and b2 == target_bit and b3 == target_bit:
                        delta += 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_pos = pos
                    
            if best_pos != -1:
                c[best_pos] = 1 - c[best_pos]
                current_score += best_delta
                improved = True
                steps += 1
                
        best_discrete_coloring = c
        best_discrete_score = current_score

    if best_discrete_coloring is None:
        # Fallback
        best_discrete_coloring = np.random.randint(0, 2, N).astype(np.int8)
        
    return best_discrete_coloring.tolist()

def construct():
    deadline = time.time() + 110 # Safety margin
    
    result = subgradient_search(deadline)
    
    return result
# EVOLVE-BLOCK-END

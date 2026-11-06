import time
import random
import math

N = 51975

def get_divisors(n):
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin
    
    divisors = get_divisors(N)
    # Filter m > 1
    divisors = [m for m in divisors if m > 1]
    k = len(divisors)
    
    # Precompute residue lists? 
    # res_lists[m_index][r] = list of x
    # This might be too much memory or time to precompute all.
    # N=51975. Sum of N/m for all m.
    # Sum_{d|N} N/d = sigma(N).
    # sigma(51975) = sigma(27)*sigma(25)*sigma(7)*sigma(11)
    # sigma(27)=1+3+9+27=40
    # sigma(25)=1+5+25=31
    # sigma(7)=8
    # sigma(11)=12
    # Product = 40 * 31 * 8 * 12 = 12160 * 12 = 145920.
    # So total elements in all residue classes is ~146k.
    # We can precompute `res_classes` as a list of lists.
    # res_classes[i] is a list of lists for divisor i.
    # res_classes[i][r] is list of x such that x % m_i == r.
    
    res_classes = []
    for m in divisors:
        classes = [[] for _ in range(m)]
        for x in range(N):
            classes[x % m].append(x)
        res_classes.append(classes)
        
    best_uncovered = N
    best_state = None
    best_cover_count = None
    
    # Greedy Initialization with multiple orders
    def run_greedy(order):
        cover_count = bytearray(N)
        uncovered = N
        state = [0] * k # dummy
        used_indices = set()
        
        # Map divisor index to position in order
        # order is a list of indices into divisors
        for idx in order:
            m = divisors[idx]
            best_r = 0
            best_gain = -1
            # We want to maximize gain (decrease in uncovered)
            # Gain = number of x in class r with cover_count[x] == 0
            for r in range(m):
                gain = 0
                for x in res_classes[idx][r]:
                    if cover_count[x] == 0:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            
            # Apply best_r
            state[idx] = best_r
            for x in res_classes[idx][best_r]:
                if cover_count[x] == 0:
                    uncovered -= 1
                cover_count[x] += 1
            
        return uncovered, state, cover_count

    # Try a few orders
    indices = list(range(k))
    
    # Order 1: Increasing m
    order1 = sorted(indices, key=lambda i: divisors[i])
    u1, s1, c1 = run_greedy(order1)
    if u1 < best_uncovered:
        best_uncovered = u1
        best_state = s1
        best_cover_count = c1
        
    # Order 2: Decreasing m
    order2 = sorted(indices, key=lambda i: divisors[i], reverse=True)
    u2, s2, c2 = run_greedy(order2)
    if u2 < best_uncovered:
        best_uncovered = u2
        best_state = s2
        best_cover_count = c2

    # Order 3: Random
    random.seed(42)
    for _ in range(5):
        order3 = indices.copy()
        random.shuffle(order3)
        u3, s3, c3 = run_greedy(order3)
        if u3 < best_uncovered:
            best_uncovered = u3
            best_state = s3
            best_cover_count = c3
            
    # Simulated Annealing
    if best_state is None:
        # Fallback
        best_state = [0]*k
        best_cover_count = bytearray(N)
        best_uncovered = N
        # Re-run greedy if needed, but we already did
        
    cover_count = best_cover_count[:]
    state = best_state[:]
    current_uncovered = best_uncovered
    
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            break
            
        # Pick random index
        idx = random.randint(0, k-1)
        m = divisors[idx]
        old_r = state[idx]
        
        # Pick random new residue
        if m == 1:
            new_r = 0
        else:
            new_r = random.randint(0, m-1)
            
        if new_r == old_r:
            continue
            
        # Calculate delta
        delta = 0
        # Remove old
        for x in res_classes[idx][old_r]:
            if cover_count[x] == 1:
                delta += 1
        # Add new
        for x in res_classes[idx][new_r]:
            if cover_count[x] == 0:
                delta -= 1
                
        if delta < 0:
            # Accept
            state[idx] = new_r
            for x in res_classes[idx][old_r]:
                cover_count[x] -= 1
            for x in res_classes[idx][new_r]:
                cover_count[x] += 1
            current_uncovered += delta
            
            if current_uncovered < best_uncovered:
                best_uncovered = current_uncovered
                best_state = state[:]
                # No need to copy cover_count for best tracking unless we want to restart from it?
                # We just track the score.
        else:
            # Accept with prob exp(-delta/T)
            if random.random() < math.exp(-delta / T):
                state[idx] = new_r
                for x in res_classes[idx][old_r]:
                    cover_count[x] -= 1
                for x in res_classes[idx][new_r]:
                    cover_count[x] += 1
                current_uncovered += delta
                
        T *= cooling
        
    # Construct result from best_state
    # best_state holds residues for each divisor index
    result = []
    for i in range(k):
        result.append([divisors[i], best_state[i]])
        
    return result

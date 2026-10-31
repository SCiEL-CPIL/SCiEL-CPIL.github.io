import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin against 120s/150s kill

def _compute_scores(u):
    """
    Computes |U+U| and |U-U| efficiently using FFT.
    Returns (size_sum, size_diff, max_u)
    """
    if not u:
        return 0, 0, 0
    if len(u) == 1:
        return 1, 1, max(u)
        
    M = max(u)
    if M == 0:
        return 1, 1, 0
        
    # Next power of 2 for FFT size
    N = 1
    while N < 2 * M + 1:
        N *= 2
        
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
            
    F = np.fft.fft(f)
    
    # Sum set
    conv_sum = np.fft.ifft(F * F).real
    # We only care about the range [0, 2*M]
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set
    # Autocorrelation
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # Count positive differences
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff), M

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    size_sum, size_diff, M = _compute_scores(u)
    if size_sum == 0 or M == 0:
        return 1.0 if M == 0 else 0.0
        
    ratio = size_diff / float(size_sum)
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * M + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _sa_search(L, initial_u, max_iter=5000):
    """
    Simulated Annealing search for a set U of size up to L.
    We represent U as a set of integers.
    Moves: Add a random element in [0, L-1] not in U, or remove a random element from U (not 0).
    """
    current_u = set(initial_u)
    current_score = _score(list(current_u))
    best_u = set(current_u)
    best_score = current_score
    
    T = 0.1
    T_min = 1e-4
    cooling = 0.999
    
    # Precompute list of all possible elements
    all_elements = list(range(L))
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 10.0:
            break
            
        # Generate move
        if len(current_u) < L - 1 and np.random.rand() < 0.5:
            # Add element
            x = np.random.choice(all_elements)
            if x in current_u:
                continue
            new_u = current_u | {int(x)}
        else:
            # Remove element (keep 0)
            removable = [x for x in current_u if x != 0]
            if not removable:
                continue
            x = np.random.choice(removable)
            new_u = current_u - {int(x)}
            
        # Evaluate
        new_score = _score(list(new_u))
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < math.exp(delta / T):
            current_u = new_u
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_u = set(current_u)
                
        T = max(T_min, T * cooling)
        
    return list(best_u), best_score

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    # Strategy:
    # 1. Initialize with a few diverse structural seeds (geometric progressions, random sets).
    # 2. Run Simulated Annealing on each seed to refine and expand/shrink the set.
    # 3. Return the best result.
    
    best_u = [0]
    best_s = 0.0
    
    # Define a maximum range for the set. 
    # The denominator is log(2M+1). 
    # If M is too large, the denominator grows, penalizing the score unless ratio is huge.
    # Typical good sets might have M up to a few thousand.
    # Let's try a few different "boxes" L.
    
    # Seed 1: Geometric progression based
    # 1, 2, 4, 8... grows M fast.
    # Let's try a mix of small dense sets and sparse sets.
    
    seeds = []
    
    # Seed A: Random set in [0, 100]
    seeds.append(list(np.random.choice(101, size=20, replace=False)))
    if 0 not in seeds[-1]:
        seeds[-1][0] = 0
        
    # Seed B: Arithmetic progression with step 2
    seeds.append(list(range(0, 100, 2)))
    
    # Seed C: Geometric progression 3^k
    geom = []
    x = 1
    while x < 200:
        geom.append(x)
        x *= 3
    seeds.append(geom)
    
    # Seed D: Random set in [0, 500]
    seeds.append(list(np.random.choice(501, size=50, replace=False)))
    if 0 not in seeds[-1]:
        seeds[-1][0] = 0
        
    # Seed E: "Golomb-like" random
    # Just random distinct integers
    seeds.append(list(np.random.choice(200, size=30, replace=False)))
    if 0 not in seeds[-1]:
        seeds[-1][0] = 0
        
    # Evaluate initial seeds
    for seed in seeds:
        seed = sorted(set(seed))
        if 0 not in seed:
            seed.append(0)
        s = _score(seed)
        if s > best_s:
            best_s = s
            best_u = seed
            
    # Run SA on the best seed and a couple of others
    # We will run SA for a limited number of iterations to stay within time.
    
    # Determine L for SA. 
    # If we start with a set with max M, we might want to allow growth up to 2*M or so.
    # But changing L mid-optimization is tricky. 
    # Let's fix L for each SA run based on the seed's max value.
    
    # Run 1: On the best seed
    if best_u:
        M_init = max(best_u)
        # Allow some growth, but not too much. 
        # If M_init is small, L can be larger to allow exploration.
        # If M_init is large, L should be close to M_init to avoid huge FFTs.
        L = max(100, int(1.5 * M_init) + 10)
        
        # Limit iterations based on time
        # Each iteration is ~1 FFT. 
        # FFT size ~ 4*L. 
        # If L=100, FFT size 400. Very fast.
        # If L=1000, FFT size 4000. Still fast.
        # Let's do 2000 iterations.
        u_sa, s_sa = _sa_search(L, best_u, max_iter=2000)
        if s_sa > best_s:
            best_s = s_sa
            best_u = u_sa
            
    # Run 2: On a different seed (e.g., the random one in [0,500])
    # Find the seed with max value around 500
    seed_500 = [s for s in seeds if max(s) > 200]
    if seed_500:
        s_seed = seed_500[0]
        s_val = _score(s_seed)
        if s_val > 0: # valid
            L = max(500, int(1.2 * max(s_seed)))
            u_sa2, s_sa2 = _sa_search(L, s_seed, max_iter=2000)
            if s_sa2 > best_s:
                best_s = s_sa2
                best_u = u_sa2

    # Run 3: A "dense" search in a small box
    # Sometimes small M with high ratio is better.
    # Try SA in box [0, 50]
    seed_small = list(np.random.choice(51, size=15, replace=False))
    if 0 not in seed_small:
        seed_small[0] = 0
    u_sa3, s_sa3 = _sa_search(51, seed_small, max_iter=2000)
    if s_sa3 > best_s:
        best_s = s_sa3
        best_u = u_sa3

    # Final polish: Greedy addition on the best SA result
    if best_u:
        M_best = max(best_u)
        max_val = max(100, int(1.5 * M_best))
        
        # Simple greedy add
        current_u = sorted(best_u)
        current_score = best_s
        current_set = set(current_u)
        
        for _ in range(50):
            if time.time() > _DEADLINE - 2.0:
                break
                
            best_candidate = None
            best_candidate_score = current_score
            
            # Sample candidates
            candidates = []
            attempts = 0
            while len(candidates) < 50 and attempts < 200:
                x = np.random.randint(1, max_val + 1)
                if x not in current_set:
                    candidates.append(x)
                attempts += 1
            
            for x in candidates:
                new_u = current_u + [x]
                new_u.sort()
                s = _score(new_u)
                if s > best_candidate_score:
                    best_candidate_score = s
                    best_candidate = x
                    
            if best_candidate is not None:
                current_u.append(best_candidate)
                current_u.sort()
                current_set.add(best_candidate)
                current_score = best_candidate_score
            else:
                break
                
        if current_score > best_s:
            best_s = current_score
            best_u = current_u

    # Ensure 0 is in best_u
    if 0 not in best_u:
        best_u.append(0)
    best_u = sorted(set(int(x) for x in best_u))
        
    return best_u

if __name__ == "__main__":
    res = construct()
    print(res)
    print(_score(res))

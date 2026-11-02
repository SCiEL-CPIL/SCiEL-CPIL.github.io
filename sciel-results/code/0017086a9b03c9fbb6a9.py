import time
import numpy as np
from scipy.optimize import minimize
from sympy import isprime, nextprime
import itertools

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Strategy:
    # We model U as a mixed-radix set: U = { sum_{i=0}^{K-1} d_i M_i }
    # where M_0 = 1, M_{i+1} = 2 * max(U_i) + 1 (gating to prevent carry/overlap).
    # This ensures |U+U| = prod |D_i+D_i| and |U-U| = prod |D_i-D_i|.
    # Score = 1 + (sum log|D_i-D_i| - sum log|D_i+D_i|) / log(2 max(U) + 1).
    
    # We optimize the digit sets D_i.
    # Instead of discrete search, we use a continuous surrogate.
    # However, the sets D_i must be discrete integers for the factorization to hold exactly.
    # But we can optimize the "structure" of D_i.
    
    # A better approach for continuous optimization:
    # Fix the number of levels K.
    # For each level, choose a small set of integers D_i.
    # We can parameterize D_i by its elements.
    # Since we need |D_i-D_i|/|D_i+D_i| > 1, we know D_i cannot be an arithmetic progression.
    
    # Let's use a hybrid:
    # 1. Define a family of "good" atoms (small sets with high ratio).
    # 2. Use continuous optimization to find the best sequence of atoms? No, that's discrete.
    
    # Alternative: Direct continuous optimization of a smooth surrogate for the score?
    # The score function is piecewise constant (step function) w.r.t the set elements.
    # Gradient is zero almost everywhere.
    
    # So, we must use a discrete search, but make it MUCH faster by using the analytic formula.
    # The current program uses FFT for every evaluation, which is slow.
    # If we enforce the gating condition M_{i+1} > 2*max(U_i), the factorization holds.
    # Then Score can be computed in O(K * |D|^2) which is very fast for small D.
    
    # Let's refine the "Atom" search to be exhaustive and fast, then do a greedy/local search
    # using the FAST analytic score.
    
    # 1. Generate a pool of "Atoms" (small sets S containing 0, sorted, max <= 20).
    # Calculate their intrinsic ratio r_S = |S-S|/|S+S| and span s_S = max(S).
    # Also calculate the "cost" in terms of max(U) growth.
    
    atoms = []
    max_span_atom = 25
    from itertools import combinations
    
    # We want to find atoms that maximize log(r) / log(span) roughly, 
    # but actually we want to maximize the contribution to the numerator per unit of log(maxU).
    
    for size in range(2, 6):
        for comb in combinations(range(1, max_span_atom + 1), size - 1):
            S = [0] + list(comb)
            S.sort()
            # Calculate ratio
            sumset = set()
            diffset = set()
            for a in S:
                for b in S:
                    sumset.add(a + b)
                    diffset.add(a - b)
            r = len(diffset) / len(sumset)
            if r > 1.0:
                atoms.append({
                    'set': S,
                    'r': r,
                    'span': max(S),
                    'log_r': np.log(r),
                    'log_span': np.log(max(S))
                })
                
    # Sort by efficiency
    atoms.sort(key=lambda x: x['log_r'] / x['log_span'], reverse=True)
    
    # 2. Greedy Construction with Local Search
    # State: List of atom indices.
    # Score function using analytic formula:
    # U is built level by level.
    # Max(U) = sum_{i=0}^{K-1} max(D_i) * M_i.
    # M_0 = 1.
    # M_{i+1} = 2 * Max(U_i) + 1.
    # Log_MaxU = log(2 * Max(U) + 1).
    # Log_Ratio = sum_{i=0}^{K-1} (log|D_i-D_i| - log|D_i+D_i|).
    
    def calc_analytic_score(atom_indices):
        if not atom_indices:
            return 0.0
        log_num = 0.0
        max_u = 0
        m = 1
        for idx in atom_indices:
            S = atoms[idx]['set']
            # Precomputed log_r is log(|S-S|/|S+S|)
            log_num += atoms[idx]['log_r']
            
            # Update Max(U)
            # Current Max(U) is max_u.
            # New Max(U) = max_u + max(S) * m
            new_max_u = max_u + max(S) * m
            max_u = new_max_u
            
            # Update M for next level
            m = 2 * max_u + 1
            
            # Safety check for overflow or excessive size
            if max_u > 10**18:
                return -np.inf
                
        if max_u == 0:
            return 0.0
        log_denom = np.log(2 * max_u + 1)
        if log_denom == 0:
            return 0.0
            
        return 1.0 + log_num / log_denom

    # Initialize best
    best_score = 0.0
    best_seq = []
    
    # Start with a few good atoms
    # Try all single atoms
    for i, a in enumerate(atoms[:50]):
        s = calc_analytic_score([i])
        if s > best_score:
            best_score = s
            best_seq = [i]
            
    # Greedy expansion
    # Add atoms that improve the score the most.
    # Since the score converges to the efficiency of the atoms, 
    # mixing different atoms might help if their efficiencies are close but one has better "tail" behavior?
    # Actually, for large K, the score is dominated by the average efficiency.
    # So we should mostly use the most efficient atom.
    # But maybe a sequence of different atoms yields a better finite-K score?
    
    # Let's do a local search on the sequence.
    # Moves:
    # 1. Replace an atom
    # 2. Add an atom
    # 3. Remove an atom
    
    current_seq = best_seq[:]
    current_score = best_score
    
    # Use the top 20 atoms for search to keep it fast
    search_pool = list(range(min(20, len(atoms))))
    
    time_limit = deadline
    step = 0
    while time.time() < time_limit:
        step += 1
        improved = False
        
        # 1. Replace
        if len(current_seq) > 0:
            idx = np.random.randint(0, len(current_seq))
            new_atom_idx = search_pool[np.random.randint(0, len(search_pool))]
            new_seq = current_seq[:]
            new_seq[idx] = new_atom_idx
            s = calc_analytic_score(new_seq)
            if s > current_score:
                current_seq = new_seq
                current_score = s
                if s > best_score:
                    best_score = s
                    best_seq = new_seq[:]
                improved = True
        
        # 2. Add
        if len(current_seq) < 30: # Cap depth to avoid huge max_u
            new_atom_idx = search_pool[np.random.randint(0, len(search_pool))]
            new_seq = current_seq + [new_atom_idx]
            s = calc_analytic_score(new_seq)
            if s > current_score:
                current_seq = new_seq
                current_score = s
                if s > best_score:
                    best_score = s
                    best_seq = new_seq[:]
                improved = True
        
        # 3. Remove
        if len(current_seq) > 1:
            new_seq = current_seq[:-1]
            s = calc_analytic_score(new_seq)
            if s > current_score:
                current_seq = new_seq
                current_score = s
                if s > best_score:
                    best_score = s
                    best_seq = new_seq[:]
                improved = True
        
        # 4. Random Restart / Perturbation if stuck
        if not improved and step % 100 == 0:
             # Try a random short sequence from the top pool
             L = np.random.randint(2, 10)
             current_seq = [search_pool[np.random.randint(0, len(search_pool))] for _ in range(L)]
             current_score = calc_analytic_score(current_seq)
             if current_score > best_score:
                 best_score = current_score
                 best_seq = current_seq[:]
        
        # 5. Simulated Annealing style acceptance? 
        # Let's just accept if better. If we get stuck, the restart handles it.
        
    # Construct the final set from best_seq
    if not best_seq:
        return [0, 1]
        
    U = [0]
    M = 1
    for idx in best_seq:
        S = atoms[idx]['set']
        new_U = []
        for u in U:
            for d in S:
                new_U.append(u + d * M)
        U = new_U
        if not U:
            break
        M = 2 * max(U) + 1
        # Check if we are running out of time or memory
        if len(U) > 100000:
            break
        if time.time() > deadline - 5:
            break
            
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")

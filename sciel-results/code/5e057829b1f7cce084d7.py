import time
import random
import numpy as np

def compute_metrics(A):
    """
    Compute sizes of A+A and A*A using NumPy for speed.
    A is a list of distinct positive integers.
    """
    n = len(A)
    # Use int64 to avoid overflow in products (max 10000*10000 = 10^8, fits in int64)
    arr = np.array(A, dtype=np.int64)
    
    # Generate indices for upper triangle including diagonal
    i, j = np.triu_indices(n)
    
    sums = arr[i] + arr[j]
    prods = arr[i] * arr[j]
    
    # Count unique elements
    # np.unique returns sorted unique values, so len gives the count
    unique_sums = np.unique(sums)
    unique_prods = np.unique(prods)
    
    return len(unique_sums), len(unique_prods)

def objective(A):
    s, p = compute_metrics(A)
    return max(s, p)

def construct():
    start_time = time.time()
    # Evaluator kills at 90s. Use 50s to be safe, leaving buffer for import/overhead.
    deadline = start_time + 50.0
    
    random.seed(42)
    
    # Helper to validate and get objective
    def get_obj(A):
        if len(set(A)) != 20:
            return float('inf')
        if any(x < 1 or x > 10000 for x in A):
            return float('inf')
        return objective(A)

    # 1. Generate diverse initial candidates
    
    candidates = []
    
    # Candidate A: Arithmetic Progression 1..20
    # |A+A| is small (39). |A*A| is large.
    ap = list(range(1, 21))
    candidates.append(ap)
    
    # Candidate B: Geometric Progression (powers of 2)
    # |A*A| is small (210 pairs, but many collisions? No, powers of 2 have unique products 2^(i+j)).
    # Wait, if A = {2^0, ..., 2^19}, products are 2^(i+j). i,j in 0..19.
    # i+j ranges from 0 to 38. So |A*A| = 39.
    # |A+A| for powers of 2: sums are unique? 2^i + 2^j.
    # If i != j, sum is unique binary representation with two 1s.
    # If i == j, sum is 2^(i+1).
    # So |A+A| is also small.
    # Let's check bounds: 2^19 = 524288 > 10000.
    # So we can't use pure powers of 2 up to 2^19.
    # Max power of 2 <= 10000 is 2^13 = 8192.
    # We need 20 elements.
    # Let's try a "compressed" geometric progression or mixed.
    
    # Candidate C: Geometric with base 1.2 or 1.3, adjusted to fit 1..10000
    # Let's try to construct a set that balances sum and product.
    # A set like {1, 2, 3, ..., 10} union {some larger numbers}
    
    # Let's generate some random "structured" sets
    for _ in range(20):
        # Random geometric-ish
        base = random.uniform(1.01, 1.15)
        start_val = random.randint(1, 5)
        A = []
        v = float(start_val)
        for _ in range(20):
            A.append(int(v))
            v *= base
            # Add noise
            v *= random.uniform(0.95, 1.05)
        
        # Ensure distinct and within bounds
        A = sorted(set(A))
        # If we lost elements due to dedup or bounds, fill them
        if len(A) < 20:
            # Fill with random values not in A
            while len(A) < 20:
                x = random.randint(1, 10000)
                if x not in A:
                    A.append(x)
            A = sorted(A)
        else:
            # If we have >20 (shouldn't happen if we append exactly 20), truncate? 
            # No, we appended 20 times. But dedup might reduce count.
            # If count < 20, we filled.
            pass
            
        # Validate bounds
        A = [x for x in A if 1 <= x <= 10000]
        if len(A) < 20:
             # Re-fill if bounds cut too many
             while len(A) < 20:
                 x = random.randint(1, 10000)
                 if x not in A:
                     A.append(x)
             A = sorted(A)
        else:
             A = A[:20] # Just in case, though logic above ensures 20
            
        if len(set(A)) == 20:
            candidates.append(A)

    # Candidate D: "Optimal" known heuristics
    # Sets that are "multiplicatively Sidon" or similar?
    # Let's try a set that is an AP in log space (GP) but clamped.
    # Or a set that is an AP in linear space (AP).
    
    # Evaluate all candidates and pick the best few as starting points for local search
    scored_candidates = []
    for A in candidates:
        obj = get_obj(A)
        if obj != float('inf'):
            scored_candidates.append((obj, A))
            
    if not scored_candidates:
        # Fallback
        return list(range(1, 21))
        
    # Sort by objective
    scored_candidates.sort(key=lambda x: x[0])
    
    best_obj = scored_candidates[0][0]
    best_set = scored_candidates[0][1][:]
    
    # Local Search Function
    def local_search(start_A, time_limit):
        nonlocal best_obj, best_set
        
        A = start_A[:]
        cur_obj = get_obj(A)
        
        if cur_obj < best_obj:
            best_obj = cur_obj
            best_set = A[:]
            
        arr = np.array(A, dtype=np.int64)
        i_idx, j_idx = np.triu_indices(20)
        
        # Precompute indices for fast update? 
        # Updating one element A[k] affects pairs (i, k) and (k, j) where i<k<j or similar.
        # Actually, A+A and A*A depend on all pairs.
        # With n=20, full recomputation is ~210 ops. NumPy overhead is small.
        # Let's stick to full recomputation for simplicity and correctness, as it's fast enough.
        
        stuck_count = 0
        start_ls_time = time.time()
        
        while time.time() < deadline and (time.time() - start_ls_time) < time_limit:
            # Generate a move
            # Pick an index to change
            idx = random.randint(0, 19)
            old_val = A[idx]
            
            # Generate candidates for new value
            cands = set()
            
            # 1. Local perturbations
            for delta in [-5, -3, -2, -1, 1, 2, 3, 5, 10, -10, 20, -20]:
                nv = old_val + delta
                if 1 <= nv <= 10000:
                    cands.add(nv)
            
            # 2. Geometric neighbors
            for factor in [0.8, 0.9, 1.1, 1.2, 1.5, 2.0, 0.5]:
                nv = int(round(old_val * factor))
                if 1 <= nv <= 10000:
                    cands.add(nv)
            
            # 3. Random
            cands.add(random.randint(1, 10000))
            
            # Filter valid candidates (distinct from other elements)
            other_vals = set(A)
            other_vals.remove(old_val)
            valid_cands = [c for c in cands if c not in other_vals and 1 <= c <= 10000]
            
            if not valid_cands:
                continue
                
            new_val = random.choice(valid_cands)
            
            # Apply move
            A[idx] = new_val
            
            # Evaluate
            new_obj = get_obj(A)
            
            if new_obj < cur_obj:
                cur_obj = new_obj
                stuck_count = 0
                if cur_obj < best_obj:
                    best_obj = cur_obj
                    best_set = A[:]
            else:
                # Revert
                A[idx] = old_val
                stuck_count += 1
                
                # If stuck for too long, do a random restart or larger jump?
                # Simple strategy: if stuck for 1000 steps, break and try next start point
                if stuck_count > 1000:
                    break

    # Run local search from top 5 distinct starting points
    # To avoid redundant work, ensure start points are distinct
    unique_starts = []
    for obj, A in scored_candidates:
        if len(unique_starts) >= 5:
            break
        # Check if similar to existing starts? Just take top 5 distinct lists
        if A not in unique_starts:
            unique_starts.append(A)
            
    for start_A in unique_starts:
        if time.time() > deadline - 5: # Keep 5s buffer
            break
        local_search(start_A, time_limit=10.0) # Allow up to 10s per restart
        
    return best_set

if __name__ == "__main__":
    # Quick self-test
    A = construct()
    print(f"Result: {A}")
    print(f"Objective: {objective(A)}")
    s, p = compute_metrics(A)
    print(f"|A+A|: {s}, |A*A|: {p}")

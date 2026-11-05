import random
import time

N = 40

def construct():
    start_time = time.time()
    timeout = 85.0
    
    target_diffs = set(range(1, 39))
    num_targets = len(target_diffs)
    
    best_S = None
    best_size = 99
    
    def get_coverage(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return len(diffs), diffs

    def is_valid(S):
        S_sorted = sorted(S)
        diffs = set()
        for i in range(len(S_sorted)):
            for j in range(i+1, len(S_sorted)):
                d = S_sorted[j] - S_sorted[i]
                if d in target_diffs:
                    diffs.add(d)
        return diffs == target_diffs

    # Size 9 is impossible (max 36 diffs < 38 required)
    # Focus on Size 10
    # If Size 10 fails, fallback to Size 11 (known to work)
    
    found_size_10 = False
    found_size_11 = False
    
    # --- Search for Size 10 ---
    # Use multiple restarts with better initialization
    
    for restart in range(100):
        if time.time() - start_time > timeout - 10:
            break
            
        # Greedy Initialization for S of size 10
        # Start with endpoints to cover large diffs? Or spread out?
        # Let's try a greedy approach:
        # 1. Pick 1 and 39.
        # 2. Iteratively pick the next point that covers the most new differences.
        
        S = [1, 39]
        covered = set()
        # Calculate initial coverage
        covered.add(38)
        
        for _ in range(8): # Need 8 more points
            best_candidate = None
            best_new_coverage = -1
            
            # Candidates: 2..38
            for c in range(2, 39):
                if c in S:
                    continue
                # Calculate new diffs with existing S
                new_diffs = set()
                for s in S:
                    d = abs(c - s)
                    if d in target_diffs and d not in covered:
                        new_diffs.add(d)
                
                if len(new_diffs) > best_new_coverage:
                    best_new_coverage = len(new_diffs)
                    best_candidate = c
            
            if best_candidate is None:
                break
            S.append(best_candidate)
            # Update covered
            for s in S[:-1]: # All except the new one
                d = abs(best_candidate - s)
                if d in target_diffs:
                    covered.add(d)
        
        # Now we have a greedy S of size 10.
        # Refine with Local Search
        current_score, _ = get_coverage(S)
        
        # Simulated Annealing / Local Search
        T = 5.0
        cooling = 0.999
        for iter in range(50000):
            if time.time() - start_time > timeout - 5:
                break
                
            if current_score == num_targets:
                best_S = S
                found_size_10 = True
                break
            
            # Move: Swap a point with a random other point
            idx = random.randint(0, len(S)-1)
            old_val = S[idx]
            new_val = random.randint(1, 39)
            
            if new_val in S and new_val != old_val:
                continue
            
            S[idx] = new_val
            new_score, _ = get_coverage(S)
            
            if new_score >= current_score:
                current_score = new_score
            else:
                delta = current_score - new_score
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_score = new_score
                else:
                    S[idx] = old_val
            
            T *= cooling
        
        if found_size_10:
            break
    
    if not found_size_10:
        # --- Search for Size 11 ---
        # We know size 11 works. Use the previous logic but optimized.
        for restart in range(20):
            if time.time() - start_time > timeout - 2:
                break
                
            S = []
            S.append(1)
            S.append(39)
            candidates = list(range(2, 39))
            random.shuffle(candidates)
            S.extend(candidates[:9])
            random.shuffle(S)
            
            current_score, _ = get_coverage(S)
            
            T = 5.0
            cooling = 0.999
            for iter in range(50000):
                if time.time() - start_time > timeout - 2:
                    break
                    
                if current_score == num_targets:
                    best_S = S
                    found_size_11 = True
                    break
                    
                idx = random.randint(0, len(S)-1)
                old_val = S[idx]
                new_val = random.randint(1, 39)
                
                if new_val in S and new_val != old_val:
                    continue
                    
                S[idx] = new_val
                new_score, _ = get_coverage(S)
                
                if new_score >= current_score:
                    current_score = new_score
                else:
                    delta = current_score - new_score
                    if random.random() < (2.718281828 ** (-delta / T)):
                        current_score = new_score
                    else:
                        S[idx] = old_val
                
                T *= cooling
            
            if found_size_11:
                break
        
        if not found_size_11:
            # Fallback
            edges = [(0, i) for i in range(1, N)] + [(i, i+1) for i in range(1, N-1)]
            cycles = [list(range(k)) for k in range(3, N+1)]
            return {'edges': edges, 'cycles': cycles}
            
    # Construct graph with best_S
    path_edges = [(i, i+1) for i in range(1, N-1)]
    spoke_edges = [(0, s) for s in best_S]
    
    edges = path_edges + spoke_edges
    
    S_set = set(best_S)
    cycles = []
    for L in range(3, N+1):
        diff_needed = L - 2
        found = False
        for u in best_S:
            v = u + diff_needed
            if v in S_set:
                cycle_verts = [0] + list(range(u, v+1))
                cycles.append(cycle_verts)
                found = True
                break
        if not found:
            raise Exception("Cycle not found for length " + str(L))
            
    return {'edges': edges, 'cycles': cycles}

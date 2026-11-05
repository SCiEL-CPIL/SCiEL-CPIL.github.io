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

    # Search for Size 10 first, then Size 11
    found_size_10 = False
    found_size_11 = False
    
    # --- Search for Size 10 ---
    for restart in range(200):
        if time.time() - start_time > timeout - 10:
            break
            
        # Greedy Initialization
        S = [1, 39]
        covered = set()
        covered.add(38)
        
        for _ in range(8):
            best_candidate = None
            best_new_coverage = -1
            
            for c in range(2, 39):
                if c in S:
                    continue
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
            for s in S[:-1]:
                d = abs(best_candidate - s)
                if d in target_diffs:
                    covered.add(d)
        
        current_score, _ = get_coverage(S)
        
        # Local Search with swap moves
        T = 5.0
        cooling = 0.9995
        for it in range(30000):
            if time.time() - start_time > timeout - 5:
                break
                
            if current_score == num_targets:
                best_S = S
                found_size_10 = True
                break
            
            # Swap two random elements
            i, j = random.sample(range(len(S)), 2)
            old_i, old_j = S[i], S[j]
            
            if old_i == old_j:
                continue
                
            S[i], S[j] = old_j, old_i
            new_score, _ = get_coverage(S)
            
            if new_score >= current_score:
                current_score = new_score
            else:
                delta = current_score - new_score
                if random.random() < (2.718281828 ** (-delta / T)):
                    current_score = new_score
                else:
                    S[i], S[j] = old_i, old_j
            
            T *= cooling
        
        if found_size_10:
            break
    
    if not found_size_10:
        # --- Search for Size 11 ---
        for restart in range(50):
            if time.time() - start_time > timeout - 2:
                break
                
            S = [1, 39]
            candidates = list(range(2, 39))
            random.shuffle(candidates)
            S.extend(candidates[:9])
            
            current_score, _ = get_coverage(S)
            
            T = 5.0
            cooling = 0.9995
            for it in range(30000):
                if time.time() - start_time > timeout - 2:
                    break
                    
                if current_score == num_targets:
                    best_S = S
                    found_size_11 = True
                    break
                    
                i, j = random.sample(range(len(S)), 2)
                old_i, old_j = S[i], S[j]
                
                if old_i == old_j:
                    continue
                    
                S[i], S[j] = old_j, old_i
                new_score, _ = get_coverage(S)
                
                if new_score >= current_score:
                    current_score = new_score
                else:
                    delta = current_score - new_score
                    if random.random() < (2.718281828 ** (-delta / T)):
                        current_score = new_score
                    else:
                        S[i], S[j] = old_i, old_j
                
                T *= cooling
            
            if found_size_11:
                break
        
        if not found_size_11:
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

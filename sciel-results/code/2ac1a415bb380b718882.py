import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 16
    k = 4
    universe = set(range(1, n + 1))
    
    # Precompute all 4-subsets
    all_sets = [set(c) for c in combinations(range(1, n + 1), k)]
    set_to_idx = {s: i for i, s in enumerate(all_sets)}
    idx_to_set = all_sets
    num_sets = len(all_sets)
    
    def get_forbidden_indices(A, B):
        """
        Returns list of indices of sets X such that {A, B, X} is a sunflower.
        A, B are indices.
        """
        A_set = idx_to_set[A]
        B_set = idx_to_set[B]
        I = A_set & B_set
        avail = universe - (A_set | B_set)
        need = k - len(I)
        if need < 0 or len(avail) < need:
            return []
        
        forbidden = []
        # I is a set, need to combine with subsets of avail
        I_list = list(I)
        for y in combinations(avail, need):
            x = set(I_list)
            x.update(y)
            idx = set_to_idx[x]
            forbidden.append(idx)
        return forbidden

    def greedy_construct(seed):
        random.seed(seed)
        F = []
        blocked = set()
        available = set(range(num_sets))
        
        while available:
            if time.time() > deadline:
                break
            
            if not F:
                best_c = random.choice(list(available))
            else:
                # Sample candidates
                candidates = list(available)
                random.shuffle(candidates)
                sample_size = min(50, len(candidates))
                best_c = None
                best_score = -1
                
                for c_idx in candidates[:sample_size]:
                    c_set = idx_to_set[c_idx]
                    score = 0
                    for a_idx in F:
                        a_set = idx_to_set[a_idx]
                        score += len(c_set & a_set)
                    
                    if score > best_score:
                        best_score = score
                        best_c = c_idx
                    elif score == best_score and best_c is not None and random.random() < 0.1:
                        best_c = c_idx
                        
                if best_c is None:
                    best_c = random.choice(list(available))
            
            F.append(best_c)
            available.remove(best_c)
            
            # Update blocked
            new_blocked = set()
            c_set = idx_to_set[best_c]
            for a_idx in F[:-1]:
                forb = get_forbidden_indices(best_c, a_idx)
                new_blocked.update(forb)
            
            blocked.update(new_blocked)
            available -= new_blocked
            
        return F

    best_F = []
    
    # Run multiple greedy instances
    for i in range(10):
        if time.time() > deadline:
            break
        F = greedy_construct(i)
        if len(F) > len(best_F):
            best_F = F
            
    result = [sorted(idx_to_set[i]) for i in best_F]
    return result

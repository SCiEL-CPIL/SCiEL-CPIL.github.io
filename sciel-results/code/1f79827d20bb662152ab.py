import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin for safety

    # 1. Generate STS(15) on points 0..14
    def get_sts_15():
        triples = set()
        for i in range(1, 16):
            for j in range(i + 1, 16):
                k = i ^ j
                if k > 0:
                    t = tuple(sorted((i, j, k)))
                    triples.add(t)
        result = []
        for t in triples:
            mapped = [x - 1 for x in t]
            result.append(mapped)
        return result

    initial_triples = get_sts_15()
    current_triples = initial_triples[:]
    
    def is_valid(triple, triples_list):
        t_set = set(triple)
        for t in triples_list:
            if len(t_set.intersection(t)) > 1:
                return False
        return True

    def calc_alpha(triples_list):
        n = 16
        triples_sets = [set(t) for t in triples_list]
        
        # We know alpha <= 9 from seed. We check from 9 downwards.
        # If we find an independent set of size k, return k.
        # If not, continue to k-1.
        for k in range(9, 0, -1):
            # Optimization: If k is 9, C(16,9) is 11440.
            # If k is 8, C(16,8) is 12870.
            # We can just check.
            found = False
            for subset in itertools.combinations(range(n), k):
                s = set(subset)
                is_independent = True
                for t_set in triples_sets:
                    if t_set.issubset(s):
                        is_independent = False
                        break
                if is_independent:
                    found = True
                    break
            if found:
                return k
            else:
                # No independent set of size k, try smaller
                continue
        return 1

    best_alpha = calc_alpha(current_triples)
    best_triples = current_triples[:]
    
    random.seed(42)
    
    # Local Search
    while time.time() < deadline:
        # Pick a random triple to remove
        idx = random.randint(0, len(current_triples) - 1)
        removed_triple = current_triples[idx]
        
        candidates = []
        
        # Try to form triples involving 15 using points from removed_triple
        # removed_triple is {a, b, c}
        # Pairs {a,b}, {b,c}, {a,c} are freed.
        for i in range(3):
            for j in range(i + 1, 3):
                p1 = removed_triple[i]
                p2 = removed_triple[j]
                cand = tuple(sorted((15, p1, p2)))
                candidates.append(list(cand))
        
        # Add some random candidates involving 15
        for _ in range(3):
            p1, p2 = random.sample(range(16), 2)
            if 15 in (p1, p2):
                continue
            cand = tuple(sorted((15, p1, p2)))
            candidates.append(list(cand))
            
        # Add general random candidates
        for _ in range(3):
            cand = list(random.sample(range(16), 3))
            candidates.append(cand)

        # Remove the triple temporarily
        current_triples.pop(idx)
        
        best_candidate = None
        best_candidate_alpha = best_alpha
        
        for cand in candidates:
            if time.time() > deadline - 2.0:
                break
            if is_valid(cand, current_triples):
                current_triples.append(cand)
                alpha = calc_alpha(current_triples)
                if alpha < best_candidate_alpha:
                    best_candidate_alpha = alpha
                    best_candidate = cand[:]
                current_triples.pop()
        
        if best_candidate is None:
            # Revert
            current_triples.append(removed_triple)
        else:
            # Keep the improvement
            current_triples.append(best_candidate)
            best_alpha = best_candidate_alpha
            best_triples = current_triples[:]
            
            if best_alpha <= 7:
                # We can keep searching for 6, but 7 is a significant jump.
                # Let's continue to see if we can get 6.
                pass

    return best_triples

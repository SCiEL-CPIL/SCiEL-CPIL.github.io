import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    
    def check_can_add(t, triple_set):
        """Check if adding triple t would create a K(2,2,2) with existing triples."""
        a, b, c = t
        for a2 in range(n):
            if a2 == a:
                continue
            for b2 in range(n):
                if b2 == b or b2 == a2:
                    continue
                for c2 in range(n):
                    if c2 == c or c2 == a2 or c2 == b2:
                        continue
                    if ((a, b, c2) in triple_set and
                        (a, b2, c) in triple_set and
                        (a, b2, c2) in triple_set and
                        (a2, b, c) in triple_set and
                        (a2, b, c2) in triple_set and
                        (a2, b2, c) in triple_set and
                        (a2, b2, c2) in triple_set):
                        return False
        return True
    
    def greedy_random(seed=0):
        rng = random.Random(seed)
        triples = list(all_triples)
        rng.shuffle(triples)
        result = []
        result_set = set()
        for t in triples:
            if check_can_add(t, result_set):
                result.append(t)
                result_set.add(t)
        return result
    
    # Run multiple greedy restarts
    best_result = []
    best_len = 0
    
    for seed in range(100):
        if time.time() > deadline - 15:
            break
        result = greedy_random(seed)
        if len(result) > best_len:
            best_len = len(result)
            best_result = result
    
    # Local search: try to add more triples
    if time.time() < deadline - 5:
        triple_set = set(best_result)
        rng = random.Random(999)
        missing = [t for t in all_triples if t not in triple_set]
        rng.shuffle(missing)
        
        for t in missing:
            if time.time() > deadline - 2:
                break
            if check_can_add(t, triple_set):
                best_result.append(t)
                triple_set.add(t)
    
    return [list(t) for t in best_result]

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} triples")

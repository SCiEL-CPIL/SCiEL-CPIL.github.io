import random
import time
import itertools

def construct():
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute for faster K(2,2,2) checking
    # For a triple (a,b,c), the forbidden pattern is:
    # There exist a2,b2,c2 such that all 8 combinations are in the set
    # We'll use a different approach: random greedy with restarts and local improvement
    
    best = []
    best_len = 0
    start_time = time.time()
    time_limit = 115  # Leave some margin for 120s limit
    
    random.seed(42)
    
    def creates_k222(triple, current_set):
        """Check if adding triple to current_set creates a K(2,2,2)"""
        a, b, c = triple
        # For each pair of points in the same "part", check if all 8 transversals exist
        # The 8 transversals for pairs {a,a2}, {b,b2}, {c,c2} are:
        # (a,b,c), (a,b,c2), (a,b2,c), (a,b2,c2), (a2,b,c), (a2,b,c2), (a2,b2,c), (a2,b2,c2)
        
        # We need to check: does there exist a2!=a, b2!=b, c2!=c such that
        # all 8 triples from {a,a2}x{b,b2}x{c,c2} are in current_set ∪ {triple}?
        
        # For efficiency, only check if the "complement" triples are in current_set
        # The complement of (a,b,c) in the 8-set is the other 7 triples
        
        # Iterate over possible a2, b2, c2
        for a2 in points:
            if a2 == a:
                continue
            for b2 in points:
                if b2 == b or b2 == a2:
                    continue
                for c2 in points:
                    if c2 == c or c2 == a2 or c2 == b2:
                        continue
                    # Check all 8 combinations
                    found = True
                    for i in (0, 1):
                        for j in (0, 1):
                            for k in (0, 1):
                                x = a if i == 0 else a2
                                y = b if j == 0 else b2
                                z = c if k == 0 else c2
                                t = tuple(sorted((x, y, z)))
                                if t == triple:
                                    continue  # This is the one we're adding
                                if t not in current_set:
                                    found = False
                                    break
                            if not found:
                                break
                        if not found:
                            break
                    if found:
                        return True
        return False
    
    attempts = 0
    while time.time() - start_time < time_limit and attempts < 200:
        attempts += 1
        random.shuffle(all_triples)
        current = set()
        
        # Greedy construction with random ordering
        for t in all_triples:
            if time.time() - start_time > time_limit:
                break
            if not creates_k222(t, current):
                current.add(t)
        
        if len(current) > best_len:
            best_len = len(current)
            best = list(current)
            
            # Try local improvement: try to add more triples
            # Remove some triples and try to add different ones
            for _ in range(5):
                if time.time() - start_time > time_limit:
                    break
                # Randomly remove a few triples
                to_remove = random.sample(list(current), min(3, len(current)))
                current_modified = set(current)
                for t in to_remove:
                    current_modified.discard(t)
                
                # Try to add back and new triples
                for t in all_triples:
                    if time.time() - start_time > time_limit:
                        break
                    if t not in current_modified and not creates_k222(t, current_modified):
                        current_modified.add(t)
                
                if len(current_modified) > best_len:
                    best_len = len(current_modified)
                    best = list(current_modified)
                    current = current_modified
    
    return [list(t) for t in best]

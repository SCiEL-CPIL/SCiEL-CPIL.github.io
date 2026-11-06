import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    n = 16
    all_pairs = set()
    for i in range(n):
        for j in range(i+1, n):
            all_pairs.add((i, j))
    
    def max_independent_set_size(triples):
        """Compute independence number by checking all subsets from large to small."""
        point_set = set(range(n))
        triple_sets = [set(t) for t in triples]
        
        # Check from largest possible downward
        for size in range(n, 0, -1):
            for subset in combinations(point_set, size):
                s = set(subset)
                is_independent = True
                for ts in triple_sets:
                    if ts <= s:
                        is_independent = False
                        break
                if is_independent:
                    return size
        return 0
    
    def is_linear(triples):
        """Check if any two triples share at most one point."""
        for i in range(len(triples)):
            for j in range(i+1, len(triples)):
                shared = set(triples[i]) & set(triples[j])
                if len(shared) > 1:
                    return False
        return True
    
    def score_of(triples):
        if not is_linear(triples):
            return float('inf')
        return max_independent_set_size(triples)
    
    # Backtracking construction of linear triple system
    def build_system(deadline):
        best_triples = None
        best_score = float('inf')
        
        def backtrack(used_pairs, current_triples, remaining_points):
            nonlocal best_triples, best_score
            
            if time.time() > deadline:
                return
            
            # Update best
            if len(current_triples) > 0:
                s = score_of(current_triples)
                if s < best_score:
                    best_score = s
                    best_triples = list(current_triples)
            
            # Try to add more triples
            # Generate candidate triples from remaining points
            candidates = []
            for i in range(len(remaining_points)):
                for j in range(i+1, len(remaining_points)):
                    for k in range(j+1, len(remaining_points)):
                        a, b, c = remaining_points[i], remaining_points[j], remaining_points[k]
                        # Check if this triple is linear with existing ones
                        triple = [a, b, c]
                        linear = True
                        for existing in current_triples:
                            shared = set(triple) & set(existing)
                            if len(shared) > 1:
                                linear = False
                                break
                        if linear:
                            # Check pair constraint (each pair of points in at most one triple)
                            pairs = [(a,b), (a,c), (b,c)]
                            if not any(p in used_pairs for p in pairs):
                                candidates.append(triple)
            
            if not candidates:
                return
            
            # Sort candidates by some heuristic - prioritize triples that use more "central" points
            random.shuffle(candidates)
            
            for triple in candidates:
                if time.time() > deadline:
                    return
                
                a, b, c = triple
                new_used_pairs = used_pairs | {(a,b), (a,c), (b,c)}
                new_remaining = [p for p in remaining_points if p not in (a,b,c)]
                
                backtrack(new_used_pairs, current_triples + [triple], new_remaining)
                
                # Also try not using this triple and continuing with others
                # This is implicit in the loop structure above
        
        # Initial call
        backtrack(set(), [], list(range(n)))
        return best_triples
    
    # Local search improvement
    def local_search(triples, deadline):
        if not triples:
            return triples
        
        best = list(triples)
        best_score = score_of(best)
        
        while time.time() < deadline:
            improved = False
            
            # Try adding a random triple
            remaining = set(range(n)) - set(p for t in best for p in t)
            if remaining:
                rem_list = list(remaining)
                if len(rem_list) >= 3:
                    for _ in range(100):
                        if time.time() > deadline:
                            break
                        candidate = random.sample(rem_list, 3)
                        candidate.sort()
                        new_triples = best + [candidate]
                        if is_linear(new_triples):
                            s = score_of(new_triples)
                            if s < best_score:
                                best = new_triples
                                best_score = s
                                improved = True
                                remaining = set(range(n)) - set(p for t in best for p in t)
                                rem_list = list(remaining)
                                break
            
            # Try swapping: remove one triple, add another
            if len(best) >= 2:
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(best)-1)
                    removed_triple = best[idx]
                    candidate_best = [t for i, t in enumerate(best) if i != idx]
                    
                    # Find a good replacement
                    used_points = set(p for t in candidate_best for p in t)
                    available = [p for p in range(n) if p not in used_points or p in removed_triple]
                    # Actually, we can use any points, just need linearity
                    for _ in range(20):
                        if time.time() > deadline:
                            break
                        candidate = random.sample(range(n), 3)
                        candidate.sort()
                        new_triples = candidate_best + [candidate]
                        if is_linear(new_triples):
                            s = score_of(new_triples)
                            if s < best_score:
                                best = new_triples
                                best_score = s
                                improved = True
                                break
            
            if not improved:
                # Try random restarts occasionally
                if random.random() < 0.1:
                    break
        
        return best
    
    # Main search
    best_result = None
    best_score = float('inf')
    
    # First, try backtracking
    t0 = time.time()
    result = build_system(deadline)
    if result is not None:
        s = score_of(result)
        if s < best_score:
            best_score = s
            best_result = result
    
    # Then, local search to improve
    if best_result is not None and time.time() < deadline:
        improved = local_search(best_result, deadline)
        if improved is not None:
            s = score_of(improved)
            if s < best_score:
                best_score = s
                best_result = improved
    
    # If backtracking didn't produce good results, try random construction
    if best_result is None or best_score > 5:
        for restart in range(100):
            if time.time() > deadline:
                break
            
            # Random greedy construction
            current = []
            used_pairs = set()
            remaining = list(range(n))
            random.shuffle(remaining)
            
            while remaining and time.time() < deadline:
                found = False
                for i in range(len(remaining)):
                    for j in range(i+1, len(remaining)):
                        for k in range(j+1, len(remaining)):
                            a, b, c = remaining[i], remaining[j], remaining[k]
                            triple = [a, b, c]
                            linear = True
                            for existing in current:
                                if len(set(triple) & set(existing)) > 1:
                                    linear = False
                                    break
                            if linear:
                                pairs = [(a,b), (a,c), (b,c)]
                                if not any(p in used_pairs for p in pairs):
                                    current.append(triple)
                                    used_pairs.update(pairs)
                                    new_remaining = [p for p in remaining if p not in (a,b,c)]
                                    random.shuffle(new_remaining)
                                    remaining = new_remaining
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                
                if not found:
                    break
            
            s = score_of(current)
            if s < best_score:
                best_score = s
                best_result = current
    
    # Final local search
    if best_result is not None and time.time() < deadline:
        improved = local_search(best_result, deadline)
        if improved is not None:
            s = score_of(improved)
            if s < best_score:
                best_score = s
                best_result = improved
    
    if best_result is None:
        # Fallback: return seed-like answer
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return best_result

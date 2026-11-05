import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110

    def get_independence_number(triples):
        points = set(range(16))
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                is_independent = True
                for a, b, c in triples:
                    if a in subset and b in subset and c in subset:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 0

    def is_linear(triples):
        n = len(triples)
        for i in range(n):
            for j in range(i + 1, n):
                common = set(triples[i]) & set(triples[j])
                if len(common) > 1:
                    return False
        return True

    def generate_sts_15():
        points = list(range(15))
        triples = []
        used_pairs = set()
        
        def backtrack(remaining_pairs):
            if not remaining_pairs:
                return True
            pair = remaining_pairs[0]
            remaining_pairs = remaining_pairs[1:]
            a, b = pair
            candidates = [c for c in points if c != a and c != b and 
                          (a, c) not in used_pairs and (b, c) not in used_pairs and
                          (c, a) not in used_pairs and (c, b) not in used_pairs]
            random.shuffle(candidates)
            for c in candidates:
                triples.append([a, b, c])
                used_pairs.add((a, b)); used_pairs.add((b, a))
                used_pairs.add((a, c)); used_pairs.add((c, a))
                used_pairs.add((b, c)); used_pairs.add((c, b))
                
                new_remaining = [p for p in remaining_pairs if 
                                 p[0] != c and p[1] != c and
                                 (p[0], p[1]) not in used_pairs and
                                 (p[1], p[0]) not in used_pairs]
                
                if backtrack(new_remaining):
                    return True
                
                triples.pop()
                used_pairs.discard((a, b)); used_pairs.discard((b, a))
                used_pairs.discard((a, c)); used_pairs.discard((c, a))
                used_pairs.discard((b, c)); used_pairs.discard((c, b))
            return False

        all_pairs = list(combinations(points, 2))
        random.shuffle(all_pairs)
        backtrack(all_pairs)
        return triples

    best_alpha = 16
    best_triples = []
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        random.seed(100 + restart)
        sts_15 = generate_sts_15()
        
        sts_pairs = set()
        for t in sts_15:
            for p in combinations(t, 2):
                sts_pairs.add((min(p), max(p)))
        
        available_pairs_15 = [(a, b) for a in range(15) for b in range(a+1, 15) if (a, b) not in sts_pairs]
        
        if len(available_pairs_15) >= 3:
            random.shuffle(available_pairs_15)
            used_points = set()
            chosen_pairs = []
            for p in available_pairs_15:
                if p[0] not in used_points and p[1] not in used_points:
                    chosen_pairs.append(p)
                    used_points.add(p[0])
                    used_points.add(p[1])
                    if len(chosen_pairs) == 3:
                        break
            
            if len(chosen_pairs) == 3:
                new_triples = sts_15 + [[15, p[0], p[1]] for p in chosen_pairs]
                if is_linear(new_triples):
                    alpha = get_independence_number(new_triples)
                    if alpha < best_alpha:
                        best_alpha = alpha
                        best_triples = new_triples
                    if best_alpha <= 4:
                        return best_triples

    return best_triples

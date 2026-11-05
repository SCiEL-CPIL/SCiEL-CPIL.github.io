import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    # Helper: Check if 4 points are collinear
    def are_collinear(p1, p2, p3, p4):
        def collinear(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) == (c[0]-a[0])*(b[1]-a[1])
        return collinear(p1, p2, p3) and collinear(p1, p2, p4)

    def is_valid(points):
        # No 4 collinear
        for combo in itertools.combinations(points, 4):
            if are_collinear(*combo):
                return False
        return True

    # Helper: Compute Score (Largest subset with no 3 collinear)
    def calc_score(points):
        n = len(points)
        # Precompute collinear triples
        triples = set()
        indices = list(range(n))
        for i, j, k in itertools.combinations(indices, 3):
            p1, p2, p3 = points[i], points[j], points[k]
            if (p2[0]-p1[0])*(p3[1]-p1[1]) == (p3[0]-p1[0])*(p2[1]-p1[1]):
                triples.add((i, j, k))
        
        # For each vertex, precompute the set of other vertices it forms a triple with
        # If we pick v, we cannot pick any pair (u, w) such that {v,u,w} is a triple.
        # This is equivalent to: if we pick v, then for any u already picked, w cannot be picked if {v,u,w} is a triple.
        
        # Let's use a more efficient B&B.
        # Sort vertices by number of triples they participate in (descending).
        # Vertices involved in many triples are "harder" to place.
        
        vertex_triples = [set() for _ in range(n)]
        for i, j, k in triples:
            vertex_triples[i].add(j)
            vertex_triples[i].add(k)
            vertex_triples[j].add(i)
            vertex_triples[j].add(k)
            vertex_triples[k].add(i)
            vertex_triples[k].add(j)
            
        # Actually, the constraint is on pairs.
        # Let's stick to the previous B&B but optimize the check.
        # Precompute a boolean matrix or bitmasks for "forms triple with".
        # triple_mask[v] is a bitmask of vertices w such that there exists u < v (or something) ? No.
        
        # Let's try a different B&B strategy:
        # Order vertices by degree in the "triple hypergraph".
        # Or simply: try to find a large independent set in the 3-uniform hypergraph.
        
        # The previous B&B was:
        # backtrack(idx, current_size)
        # Check if points[idx] forms a triple with any pair in `chosen`.
        
        # Optimization:
        # 1. Sort vertices by number of triples.
        # 2. Use bitmasks for `chosen`?
        # 3. Precompute for each vertex v, the list of pairs (u, w) it forms a triple with.
        
        pairs_for_vertex = [[] for _ in range(n)]
        for i, j, k in triples:
            pairs_for_vertex[i].append((j, k))
            pairs_for_vertex[j].append((i, k))
            pairs_for_vertex[k].append((i, j))
            
        # To check if v can be added to chosen:
        # We need to ensure no (u, w) in pairs_for_vertex[v] has both u and w in chosen.
        
        best_size = 0
        chosen = [False] * n
        chosen_indices = [] # Keep a list of chosen indices for fast iteration
        
        # Sort vertices by number of pairs they're involved in (descending)
        # This helps prune earlier
        order = sorted(range(n), key=lambda v: len(pairs_for_vertex[v]), reverse=True)
        
        def backtrack(idx, current_size):
            nonlocal best_size
            if idx == n:
                if current_size > best_size:
                    best_size = current_size
                return
            
            # Pruning
            if current_size + (n - idx) <= best_size:
                return
                
            v = order[idx]
            
            # Option 1: Include points[v]
            valid_inclusion = True
            # Check if any pair in chosen forms a triple with v
            for u, w in pairs_for_vertex[v]:
                if chosen[u] and chosen[w]:
                    valid_inclusion = False
                    break
                    
            if valid_inclusion:
                chosen[v] = True
                chosen_indices.append(v)
                backtrack(idx + 1, current_size + 1)
                chosen[v] = False
                chosen_indices.pop()
            
            # Option 2: Exclude points[v]
            backtrack(idx + 1, current_size)
        
        backtrack(0, 0)
        return best_size

    def random_point():
        return (random.randint(-100, 100), random.randint(-100, 100))
    
    def generate_initial():
        points = []
        while len(points) < 20:
            p = random_point()
            if p not in points:
                points.append(p)
        return points

    best_points = None
    best_score = 21 # Max possible is 20

    # Initial random search
    while time.time() < deadline:
        # Try a random valid config
        points = generate_initial()
        
        if not is_valid(points):
            continue
            
        score = calc_score(points)
        
        if score < best_score:
            best_score = score
            best_points = points
            # If we find a very low score, we can try to polish it
            if best_score <= 8:
                # Don't break immediately, maybe we can find lower?
                # But 8 is very low. Let's do a dedicated polish loop.
                pass
        
        # If we have a best, do local search on it
        if best_points is not None:
            # Local search step
            idx = random.randint(0, 19)
            old_p = best_points[idx]
            new_p = (random.randint(-100, 100), random.randint(-100, 100))
            
            if new_p in best_points:
                continue
                
            best_points[idx] = new_p
            
            if is_valid(best_points):
                new_score = calc_score(best_points)
                if new_score < best_score:
                    best_score = new_score
                else:
                    best_points[idx] = old_p
            else:
                best_points[idx] = old_p

    # Dedicated local search phase if we have a good candidate
    if best_points is not None and time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(20):
                if time.time() >= deadline:
                    break
                old_p = best_points[i]
                # Try multiple random replacements
                for _ in range(10):
                    new_p = (random.randint(-100, 100), random.randint(-100, 100))
                    if new_p in best_points:
                        continue
                    best_points[i] = new_p
                    if is_valid(best_points):
                        new_score = calc_score(best_points)
                        if new_score < best_score:
                            best_score = new_score
                            improved = True
                            break
                    best_points[i] = old_p
                else:
                    # No improvement found for this index, continue to next
                    continue
                if improved:
                    break
    
    if best_points is None:
        best_points = [[x, x*x] for x in range(-10, 10)]
        
    return [[p[0], p[1]] for p in best_points]

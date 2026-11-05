import itertools
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer
    
    N = 20
    LIMIT = 100
    
    # Helper: Check if 4 points are collinear
    def is_valid(points):
        # points is a list of (x, y)
        # Check all lines defined by pairs
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                count = 2
                for k in range(n):
                    if k == i or k == j:
                        continue
                    x3, y3 = points[k]
                    # Cross product check
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        count += 1
                        if count >= 4:
                            return False
        return True

    # Helper: Compute Score (Largest subset with no 3 collinear)
    def calc_score(points):
        n = len(points)
        # Precompute collinear triples
        # triple_mask[i] is a bitmask of j such that there exists k < i? 
        # Actually, let's just store the list of triples.
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                for k in range(j + 1, n):
                    x3, y3 = points[k]
                    if dx * (y3 - y1) == dy * (x3 - x1):
                        triples.append((i, j, k))
        
        # If no triples, score is n
        if not triples:
            return n
            
        # Build adjacency for pruning:
        # For a vertex v, and a set of chosen vertices C, 
        # v can be added if there is no u, w in C such that {v, u, w} is a triple.
        # This is equivalent to: for each u in C, v cannot be connected to u via a triple with any w in C.
        
        # Let's use a B&B with bitmasks for speed.
        # mask_all = (1 << n) - 1
        
        # Precompute for each v, a list of (u, w) pairs? No, that's O(N^3) memory which is fine for 20.
        # But checking "is there a pair in C" is O(|C|^2) or O(|C| * something).
        
        # Better: 
        # For each v, precompute `conflict_pairs[v]` as a list of bitmasks?
        # No, let's stick to sets of indices for clarity and speed is likely sufficient for N=20.
        
        # Optimize B&B:
        # Order vertices by degree in the hypergraph (number of triples they are in).
        degrees = [0] * n
        for i, j, k in triples:
            degrees[i] += 1
            degrees[j] += 1
            degrees[k] += 1
            
        order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
        
        # Map original index to position in order? 
        # Or just iterate through `order`.
        
        # Precompute triple lookup: set of frozensets or sorted tuples for O(1) check?
        # For N=20, linear search in triples list is fine if short, but set is better.
        triple_set = set(triples)
        
        # For faster check: 
        # Given v and chosen (set of indices), is there a triple containing v and two from chosen?
        # We can precompute for each v, a dict mapping u -> bitmask of w such that (v,u,w) is a triple.
        # Then for a given chosen set C (as bitmask), and v:
        # For each u in C:
        #   if triple_mask[v][u] & C != 0: conflict.
        
        # Let's build this.
        # triple_mask[v] is a dict: u -> bitmask of w
        # Since indices are 0..19, we can use int bitmasks.
        
        triple_masks = [dict() for _ in range(n)]
        for i, j, k in triples:
            # Ensure i, j, k are distinct
            if i not in triple_masks[j]: triple_masks[j][i] = 0
            triple_masks[j][i] |= (1 << k)
            triple_masks[j][i] |= (1 << i) # Wait, if j is the center? No, the triple is {i,j,k}.
            # If we pick j, and have i in chosen, we ban k.
            # If we pick j, and have k in chosen, we ban i.
            # So for vertex j, if we have i, we ban k.
            
            # Let's structure it as:
            # mask[v][u] = bitmask of w such that {v, u, w} is a triple.
            
            # Add (i, j, k)
            # For v=i, u=j, w=k
            if j not in triple_masks[i]: triple_masks[i][j] = 0
            triple_masks[i][j] |= (1 << k)
            
            # For v=i, u=k, w=j
            if k not in triple_masks[i]: triple_masks[i][k] = 0
            triple_masks[i][k] |= (1 << j)
            
            # For v=j, u=i, w=k
            if i not in triple_masks[j]: triple_masks[j][i] = 0
            triple_masks[j][i] |= (1 << k)
            
            # For v=j, u=k, w=i
            if k not in triple_masks[j]: triple_masks[j][k] = 0
            triple_masks[j][k] |= (1 << i)
            
            # For v=k, u=i, w=j
            if i not in triple_masks[k]: triple_masks[k][i] = 0
            triple_masks[k][i] |= (1 << j)
            
            # For v=k, u=j, w=i
            if j not in triple_masks[k]: triple_masks[k][j] = 0
            triple_masks[k][j] |= (1 << i)
            
        best_size = 0
        # chosen_mask: bitmask of chosen vertices
        # remaining_mask: bitmask of remaining vertices
        
        def backtrack(remaining_mask, chosen_mask, current_size):
            nonlocal best_size
            if current_size > best_size:
                best_size = current_size
                
            if remaining_mask == 0:
                return
                
            # Pruning
            # Max possible size is current_size + popcount(remaining_mask)
            if current_size + bin(remaining_mask).count('1') <= best_size:
                return
                
            # Pick a vertex from remaining
            # Heuristic: pick one with highest degree in the remaining subgraph?
            # For simplicity, pick the first one in `order` that is in remaining.
            v = -1
            for idx in order:
                if remaining_mask & (1 << idx):
                    v = idx
                    break
                    
            if v == -1:
                return
                
            bit_v = 1 << v
            
            # Option 1: Include v
            # Check if v conflicts with chosen
            valid_include = True
            # v conflicts if there exists u in chosen such that triple_masks[v][u] & chosen_mask != 0
            # We need to iterate u in chosen_mask.
            # Since N=20, we can just iterate all u? Or use bit operations.
            # Let's iterate u from 0 to n-1.
            for u in range(n):
                if chosen_mask & (1 << u):
                    if u in triple_masks[v]:
                        if triple_masks[v][u] & chosen_mask:
                            valid_include = False
                            break
                            
            if valid_include:
                new_chosen = chosen_mask | bit_v
                # Update remaining: remove v, and remove any w that conflicts with v and any u in new_chosen?
                # Wait, the conflict is: if we pick v, we cannot pick any pair (u, w) from the PREVIOUS chosen set?
                # No, the constraint is: NO THREE collinear in the FINAL set.
                # So if we pick v, we must ensure that for any u, w in the FINAL set, {v, u, w} is not a triple.
                # This means:
                # 1. No u, w in PREVIOUS chosen such that {v, u, w} is a triple. (Checked above)
                # 2. No u in PREVIOUS chosen, w in REMAINING such that {v, u, w} is a triple. -> Ban w.
                # 3. No u, w in REMAINING such that {v, u, w} is a triple. -> This will be handled in recursion?
                #    If we pick u later, it will check against v (which is now in chosen).
                
                # So we only need to ban w in remaining that form a triple with v and some u in chosen.
                banned_mask = 0
                for u in range(n):
                    if chosen_mask & (1 << u):
                        if u in triple_masks[v]:
                            banned_mask |= triple_masks[v][u]
                            
                # Also, v itself is removed from remaining
                new_remaining = (remaining_mask & ~bit_v) & ~banned_mask
                
                backtrack(new_remaining, new_chosen, current_size + 1)
            
            # Option 2: Exclude v
            new_remaining = remaining_mask & ~bit_v
            backtrack(new_remaining, chosen_mask, current_size)
            
        backtrack((1 << n) - 1, 0, 0)
        return best_size

    def random_point():
        return (random.randint(-LIMIT, LIMIT), random.randint(-LIMIT, LIMIT))
    
    def generate_initial():
        points = []
        while len(points) < N:
            p = random_point()
            if p not in points:
                points.append(p)
        return points

    best_points = None
    best_score = N + 1 # Max possible is 20
    
    # Main loop: Simulated Annealing
    # We need a valid starting point.
    # Random restarts until we find a valid one, then SA.
    
    while time.time() < deadline:
        # Find a valid starting configuration
        points = None
        for _ in range(1000):
            pts = generate_initial()
            if is_valid(pts):
                points = pts
                break
        
        if points is None:
            continue
            
        current_score = calc_score(points)
        if current_score < best_score:
            best_score = current_score
            best_points = points[:]
            
        # Simulated Annealing
        T = 1.0
        cooling = 0.9999
        steps = 0
        sa_deadline = time.time() + 10.0 # Run SA for up to 10s per restart
        
        while time.time() < sa_deadline and time.time() < deadline:
            idx = random.randint(0, N - 1)
            old_p = points[idx]
            new_p = random_point()
            
            if new_p in points:
                continue
                
            points[idx] = new_p
            
            if is_valid(points):
                new_score = calc_score(points)
                delta = new_score - current_score
                if delta < 0 or (delta == 0 and random.random() < 0.5) or (delta > 0 and random.random() < math.exp(-delta / T)):
                    current_score = new_score
                    if current_score < best_score:
                        best_score = current_score
                        best_points = points[:]
                else:
                    points[idx] = old_p
            else:
                points[idx] = old_p
                
            T *= cooling
            steps += 1
            
    if best_points is None:
        # Fallback: try to construct a known valid pattern
        # e.g. points on parabola y=x^2, but check 4-collinear?
        # Parabola y=x^2 has no 3 collinear? 
        # Line intersects parabola at max 2 points. So no 3 collinear.
        # But we need 20 points.
        # x = -10..9 -> 20 points.
        pts = [[x, x*x] for x in range(-10, 10)]
        # Check validity
        if is_valid(pts):
            best_points = pts
            best_score = calc_score(pts)
        else:
            # Just return random valid if found earlier, else fail
            return generate_initial()

    return [[p[0], p[1]] for p in best_points]

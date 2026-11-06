import time
import random
import math

N = 32
MAX_VAL = 512
TIME_LIMIT = 145.0  # Safety margin from 150s

def _compute_diff_counts(S):
    """Compute diff_counts for a set S. O(N^2) but only called at init/restart."""
    dc = [0] * (MAX_VAL + 1)
    S_set = set(S)
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            a, b = S[i], S[j]
            if a > b:
                a, b = b, a
            d = b - a
            # a is middle: a-d, a, a+d=b => check a-d
            if (a - d) in S_set:
                dc[d] += 1
            # a is start: a, a+d=b, a+2d => check a+2d = b+d
            if (b + d) in S_set:
                dc[d] += 1
            # b is end: a, c, b where c=a+d/2 => check if d even and a+d/2 in S
            if d % 2 == 0:
                d2 = d // 2
                if (a + d2) in S_set:
                    dc[d2] += 1
    return dc

def _score_from_dc(dc):
    return sum(1 for d in range(1, MAX_VAL + 1) if dc[d] > 0)

def _remove_contrib(x, S_set, dc):
    """Remove x's contributions to dc. S_set should NOT contain x yet.
    For each other y in S, find all 3-APs that include x and y, decrement dc."""
    # For each y in S_set, consider all 3-APs containing both x and y.
    # A 3-AP {p, q, r} with q-p = r-q = d has three elements.
    # If x and y are in the AP, the third element is determined.
    # Cases: x and y are (start, middle), (start, end), (middle, end)
    
    for y in S_set:
        if y == x:
            continue
        # Determine all ways x and y can be part of a 3-AP together
        # 1. x is start, y is middle: AP = (x, y, y + (y-x)), d = y - x
        d1 = y - x
        if d1 > 0:
            c = y + d1
            if c in S_set:
                dc[d1] -= 1
        # 2. x is middle, y is end: AP = (x - (y-x), x, y), d = y - x
        d2 = y - x
        if d2 > 0:
            c = x - d2
            if c in S_set:
                dc[d2] -= 1
        # 3. x is start, y is end: AP = (x, x + (y-x)//2, y), d = (y-x)//2 if even
        diff = y - x
        if diff > 0 and diff % 2 == 0:
            d3 = diff // 2
            c = x + d3
            if c in S_set:
                dc[d3] -= 1
        # 4. x is end, y is middle: AP = (y - (y-x), y, x) -> same as case 2 with x,y swapped
        #    Actually if y > x: y is end, x is middle? No, x < y.
        #    Let me reconsider. We need all unordered pairs {x,y} and all 3-APs containing both.
        #    A 3-AP has 3 elements. Given two elements, the third is determined in up to 3 ways.
        #    Let's be more careful.
        
        # Actually, let me reconsider the approach. For a given pair (x, y) with x != y,
        # the possible 3-APs containing both are:
        # - If |x-y| = d: 
        #   a) x, y, y+d (x start, y middle) -> need y+d in S
        #   b) x-d, x, y (x middle, y end) -> need x-d in S  [only if x-d >= 1]
        #   c) If (y-x) even: x, x+(y-x)/2, y (x start, y end) -> need x+(y-x)/2 in S
        #   d) If (y-x) even: y, y+(y-x)/2, ... no wait
        #   Actually for x < y:
        #   - (x, y, y + (y-x)): x start, y middle, third = y + (y-x)
        #   - (x - (y-x), x, y): x middle, y end, third = x - (y-x)
        #   - If (y-x) even: (x, x + (y-x)//2, y): x start, y end, third = x + (y-x)//2
        #   These are the only 3-APs containing both x and y when x < y.
        #   But wait, what about y being start and x being middle or end?
        #   If y > x:
        #   - y start, x middle: (y, x, x - (y-x)) -> x - (y-x) = 2x - y. Need 2x-y in S.
        #     But this is the same as case (b) above with roles swapped? No.
        #     Case (b) was x middle, y end: (x-(y-x), x, y). 
        #     y start, x middle: (y, x, 2x-y). d = y-x. Third = 2x-y.
        #     This is different from (b) which had third = x-(y-x) = 2x-y. Same!
        #     So (b) covers both "x middle, y end" AND "y start, x middle"? 
        #     No wait. In (b), the AP is (2x-y, x, y) with d = y-x. The elements are 2x-y, x, y.
        #     In "y start, x middle", the AP is (y, x, 2x-y) which is the same set {2x-y, x, y}.
        #     So yes, same AP.
        #   - y start, x end: (y, y-(y-x)/2, x) if (y-x) even. Third = y - (y-x)/2 = (x+y)/2.
        #     This is the same as case (c) with x and y swapped? Case (c) was (x, x+(y-x)/2, y) = (x, (x+y)/2, y).
        #     Same set! So (c) covers both.
        #   - y middle, x end: (y - (y-x), y, x) = (2x-y, y, x). Same as (b).
        #   - y end, x start: (x, (x+y)/2, y). Same as (c).
        #   - y start, x middle: covered by (b).
        #   - x start, y middle: covered by (a).
        #   - x middle, y end: covered by (b).
        #   - x start, y end: covered by (c).
        #   
        #   So for x < y, the distinct 3-APs containing both are:
        #   (a) {x, y, y + (y-x)} with d = y-x
        #   (b) {2x-y, x, y} with d = y-x  (if 2x-y >= 1)
        #   (c) {x, (x+y)//2, y} with d = (y-x)//2 (if (y-x) even and (x+y)//2 not equal to x or y)
        
        # Let me redo this more carefully.
        pass  # This function needs to be rewritten

def construct():
    start_time = time.time()
    
    def deadline():
        return time.time() - start_time > TIME_LIMIT
    
    best_score = -1
    best_S = None
    
    # Structured initializations
    def gen_random():
        return random.sample(range(1, MAX_VAL + 1), N)
    
    def gen_ap():
        # Arithmetic progression
        start = random.randint(1, 50)
        step = random.randint(2, 15)
        S = []
        for i in range(N):
            v = start + i * step
            if 1 <= v <= MAX_VAL:
                S.append(v)
            else:
                break
        while len(S) < N:
            v = random.randint(1, MAX_VAL)
            if v not in S:
                S.append(v)
        return S[:N]
    
    def gen_geometric_like():
        # Elements spread out with decreasing gaps
        S = []
        base = random.randint(1, 20)
        S.append(base)
        current = base
        for i in range(N - 1):
            gap = max(1, random.randint(1, MAX_VAL // (i + 2)))
            current += gap
            if current > MAX_VAL:
                current = random.randint(1, MAX_VAL)
            if current not in S:
                S.append(current)
        while len(S) < N:
            v = random.randint(1, MAX_VAL)
            if v not in S:
                S.append(v)
        return S[:N]
    
    def gen_clustered():
        # A few clusters
        S = []
        num_clusters = random.randint(3, 6)
        cluster_sizes = [N // num_clusters + (1 if i < N % num_clusters else 0) for i in range(num_clusters)]
        for size in cluster_sizes:
            center = random.randint(10, MAX_VAL - 10)
            for _ in range(size):
                v = center + random.randint(-15, 15)
                v = max(1, min(MAX_VAL, v))
                if v not in S:
                    S.append(v)
        while len(S) < N:
            v = random.randint(1, MAX_VAL)
            if v not in S:
                S.append(v)
        return S[:N]
    
    def gen_mixed():
        # Mix of AP and random
        S = gen_ap()
        for i in range(N // 4):
            S[i] = random.randint(1, MAX_VAL)
        # Remove duplicates
        seen = set()
        unique = []
        for v in S:
            if v not in seen and 1 <= v <= MAX_VAL:
                seen.add(v)
                unique.append(v)
        while len(unique) < N:
            v = random.randint(1, MAX_VAL)
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique[:N]
    
    generators = [gen_random, gen_ap, gen_geometric_like, gen_clustered, gen_mixed]
    
    def sa_run(initial_S):
        """Run SA on a copy of initial_S, return (best_score, best_S) found during this run."""
        S = list(initial_S)
        S_set = set(S)
        dc = _compute_diff_counts(S)
        current_score = _score_from_dc(dc)
        
        local_best = current_score
        local_best_S = list(S)
        
        T = 8.0
        T_min = 0.05
        alpha = 0.995
        
        while not deadline():
            # Pick random element to remove
            idx = random.randint(0, N - 1)
            x_out = S[idx]
            
            # Pick random element to add
            x_in = random.randint(1, MAX_VAL)
            if x_in in S_set:
                continue
            
            # Compute delta score by tentatively removing x_out and adding x_in
            # Remove x_out's contributions
            old_dc = dc  # We'll modify in place and revert if needed
            
            # For each y in S_set (excluding x_out), compute contributions of (x_out, y)
            # and remove them
            S_set_no_out = S_set - {x_out}
            
            delta_score = 0
            
            # Remove x_out contributions
            for y in S_set_no_out:
                x, yv = x_out, y
                if x > yv:
                    x, yv = yv, x
                d = yv - x
                # (a) x start, y middle: third = yv + d
                c = yv + d
                if c in S_set_no_out:
                    old_dc[d] -= 1
                    if old_dc[d] == 0:
                        delta_score -= 1
                # (b) x middle, y end: third = x - d
                c = x - d
                if c >= 1 and c in S_set_no_out:
                    old_dc[d] -= 1
                    if old_dc[d] == 0:
                        delta_score -= 1
                # (c) x start, y end: d2 = d//2, third = x + d2
                if d % 2 == 0:
                    d2 = d // 2
                    c = x + d2
                    if c in S_set_no_out:
                        old_dc[d2] -= 1
                        if old_dc[d2] == 0:
                            delta_score -= 1
            
            # Add x_in contributions
            S_set_no_out.add(x_in)
            for y in S_set_no_out:
                if y == x_in:
                    continue
                x, yv = x_in, y
                if x > yv:
                    x, yv = yv, x
                d = yv - x
                # (a) x start, y middle: third = yv + d
                c = yv + d
                if c in S_set_no_out:
                    old_dc[d] += 1
                    if old_dc[d] == 1:
                        delta_score += 1
                # (b) x middle, y end: third = x - d
                c = x - d
                if c >= 1 and c in S_set_no_out:
                    old_dc[d] += 1
                    if old_dc[d] == 1:
                        delta_score += 1
                # (c) x start, y end: d2 = d//2, third = x + d2
                if d % 2 == 0:
                    d2 = d // 2
                    c = x + d2
                    if c in S_set_no_out:
                        old_dc[d2] += 1
                        if old_dc[d2] == 1:
                            delta_score += 1
            
            # Accept/reject
            if delta_score >= 0:
                # Accept
                S[idx] = x_in
                S_set.remove(x_out)
                S_set.add(x_in)
                current_score += delta_score
                if current_score > local_best:
                    local_best = current_score
                    local_best_S = list(S)
            else:
                # Metropolis
                accept_prob = math.exp(delta_score / T)
                if random.random() < accept_prob:
                    S[idx] = x_in
                    S_set.remove(x_out)
                    S_set.add(x_in)
                    current_score += delta_score
                # else: reject, dc unchanged
            
            T *= alpha
            if T < T_min:
                break
        
        return local_best, local_best_S
    
    # Multi-restart SA with diverse initializations
    num_restarts = 8
    for r in range(num_restarts):
        if deadline():
            break
        
        gen = generators[r % len(generators)]
        init_S = gen()
        
        # Quick greedy polish on the initial set
        S_set = set(init_S)
        dc = _compute_diff_counts(init_S)
        cur_score = _score_from_dc(dc)
        
        # Greedy polish: try swapping each element with a better one
        improved = True
        polish_iter = 0
        while improved and not deadline() and polish_iter < 50:
            improved = False
            polish_iter += 1
            for i in range(N):
                x_out = init_S[i]
                best_gain = 0
                best_x_in = None
                # Try a few random candidates
                for _ in range(10):
                    x_in = random.randint(1, MAX_VAL)
                    if x_in in S_set:
                        continue
                    # Quick score estimate
                    new_S = list(init_S)
                    new_S[i] = x_in
                    new_dc = _compute_diff_counts(new_S)
                    new_score = _score_from_dc(new_dc)
                    if new_score - cur_score > best_gain:
                        best_gain = new_score - cur_score
                        best_x_in = x_in
                if best_x_in is not None:
                    S_set.remove(x_out)
                    S_set.add(best_x_in)
                    init_S[i] = best_x_in
                    cur_score += best_gain
                    improved = True
                    dc = _compute_diff_counts(init_S)
                    cur_score = _score_from_dc(dc)
        
        if cur_score > best_score:
            best_score = cur_score
            best_S = list(init_S)
        
        # SA polish
        sa_best, sa_best_S = sa_run(init_S)
        if sa_best > best_score:
            best_score = sa_best
            best_S = sa_best_S
    
    if best_S is None:
        best_S = list(range(1, N + 1))
    
    return best_S

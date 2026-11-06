import time
import random
from math import gcd

DEADLINE = 115.0  # Safety margin
N = 20
MAXC = 100

def _line_key(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    g = gcd(abs(dx), abs(dy))
    if g == 0:
        return (0, 0, 0)
    dx //= g
    dy //= g
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    a = -dy
    b = dx
    c = dy * x1 - dx * y1
    g2 = gcd(abs(a), gcd(abs(b), abs(c)))
    if g2 != 0:
        a //= g2
        b //= g2
        c //= g2
    return (a, b, c)

def _get_lines_and_triples(pts):
    """
    Returns:
    lines: dict mapping line_key -> set of indices
    triples: set of frozensets of 3 indices that are collinear
    max_line_size: maximum number of points on any line
    """
    lines = {}
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            k = _line_key(pts[i], pts[j])
            if k not in lines:
                lines[k] = set()
            lines[k].add(i)
            lines[k].add(j)
    
    triples = set()
    max_line_size = 0
    for key, idx_set in lines.items():
        sz = len(idx_set)
        if sz > max_line_size:
            max_line_size = sz
        if sz >= 3:
            idx_list = sorted(idx_set)
            for a in range(len(idx_list)):
                for b in range(a + 1, len(idx_list)):
                    for c in range(b + 1, len(idx_list)):
                        triples.add(frozenset((idx_list[a], idx_list[b], idx_list[c])))
    return lines, triples, max_line_size

def _min_hitting_set_bnb(edges):
    """
    Finds the minimum number of vertices to remove to hit all edges in the 3-uniform hypergraph.
    This is equivalent to N - max_independent_set_size.
    """
    if not edges:
        return 0
    
    best = [N] # Max possible hitting set size is N
    
    # Pre-compute edge membership for fast filtering? 
    # Since N is small, iterating is fine.
    
    def bnb(remaining_edges, current_count):
        if current_count >= best[0]:
            return
        
        if not remaining_edges:
            if current_count < best[0]:
                best[0] = current_count
            return
        
        # Lower bound: size of a maximal matching of disjoint edges
        # Greedy matching
        matched = set()
        matching_size = 0
        # Sort edges to make it deterministic and potentially better
        sorted_edges = sorted(remaining_edges, key=lambda x: tuple(x))
        for e in sorted_edges:
            if not (e & matched):
                matching_size += 1
                matched.update(e)
        
        if current_count + matching_size >= best[0]:
            return
            
        # Pick an edge to branch on
        # Heuristic: pick edge with smallest "impact" or just first?
        # Let's pick the edge that appears in the most other edges? 
        # Or just the first one for simplicity, as matching bound is strong.
        e = sorted_edges[0]
        
        for v in e:
            # New remaining edges: those that do NOT contain v
            new_edges = set()
            for ed in remaining_edges:
                if v not in ed:
                    new_edges.add(ed)
            bnb(new_edges, current_count + 1)
            
    bnb(set(edges), 0)
    return best[0]

def _score(pts):
    _, triples, _ = _get_lines_and_triples(pts)
    if not triples:
        return N
    min_hits = _min_hitting_set_bnb(triples)
    return N - min_hits

def _count_3_lines(pts):
    lines, _, max_sz = _get_lines_and_triples(pts)
    count = 0
    for idx_set in lines.values():
        if len(idx_set) == 3:
            count += 1
    return count, max_sz

def _is_valid(pts):
    _, _, max_sz = _get_lines_and_triples(pts)
    return max_sz <= 3

def _generate_initial():
    # Start with a grid-like structure
    pts = []
    # 5x4 grid
    for i in range(5):
        for j in range(4):
            x = i * 20 - 40
            y = j * 20 - 30
            # Add small perturbation to break symmetries and 4-collinear
            px = x + random.randint(-2, 2)
            py = y + random.randint(-2, 2)
            # Clamp
            if px > MAXC: px = MAXC
            if px < -MAXC: px = -MAXC
            if py > MAXC: py = MAXC
            if py < -MAXC: py = -MAXC
            pts.append([px, py])
    return pts

def construct():
    t0 = time.time()
    random.seed(42)
    
    # Initial valid points
    best_pts = None
    best_score = N
    
    # Try a few initial structures
    initial_candidates = []
    
    # 1. Perturbed Grid
    p1 = _generate_initial()
    # Ensure distinctness
    seen = set()
    ded_p1 = []
    for p in p1:
        if (p[0], p[1]) not in seen:
            seen.add((p[0], p[1]))
            ded_p1.append(p)
    while len(ded_p1) < N:
        px = random.randint(-MAXC, MAXC)
        py = random.randint(-MAXC, MAXC)
        if (px, py) not in seen:
            seen.add((px, py))
            ded_p1.append([px, py])
    initial_candidates.append(ded_p1)
    
    # 2. Parabola
    p2 = []
    for i in range(-10, 10):
        p2.append([i, i*i])
    initial_candidates.append(p2)
    
    # 3. Random
    p3 = []
    seen = set()
    while len(p3) < N:
        px = random.randint(-MAXC, MAXC)
        py = random.randint(-MAXC, MAXC)
        if (px, py) not in seen:
            seen.add((px, py))
            p3.append([px, py])
    initial_candidates.append(p3)
    
    best_pts = None
    best_score = N
    best_proxy = -1
    
    for cand in initial_candidates:
        if not _is_valid(cand):
            continue
        proxy, _ = _count_3_lines(cand)
        score = _score(cand)
        if score < best_score or (score == best_score and proxy > best_proxy):
            best_score = score
            best_pts = cand[:]
            best_proxy = proxy
            
    if best_pts is None:
        # Fallback: return any valid set (should not happen)
        return initial_candidates[0]
        
    cur_pts = best_pts[:]
    cur_score = best_score
    cur_proxy = best_proxy
    
    # Simulated Annealing
    T = 100.0
    T_min = 1.0
    alpha = 0.995
    max_steps = 10000
    
    steps = 0
    while time.time() - t0 < DEADLINE and steps < max_steps:
        steps += 1
        
        # Move: swap two points
        i, j = random.sample(range(N), 2)
        new_pts = cur_pts[:]
        new_pts[i], new_pts[j] = new_pts[j], new_pts[i]
        
        if not _is_valid(new_pts):
            continue
            
        new_proxy, _ = _count_3_lines(new_pts)
        
        # Objective: Maximize proxy (number of 3-lines)
        delta_proxy = new_proxy - cur_proxy
        
        # If proxy improves, accept.
        # If proxy worsens, accept with probability exp(delta/T)
        if delta_proxy >= 0:
            accept = True
        else:
            accept = random.random() < (2.71828 ** (delta_proxy / T))
            
        if accept:
            cur_pts = new_pts
            cur_proxy = new_proxy
            
            # Check score periodically or if proxy improved significantly
            # Score is expensive. Only compute if we might have improved best_score
            # Since score >= N - (some function of proxy), we can use a heuristic
            # But to be safe, compute score if proxy > best_proxy
            if cur_proxy > best_proxy:
                new_score = _score(cur_pts)
                if new_score < best_score:
                    best_score = new_score
                    best_pts = cur_pts[:]
                    best_proxy = cur_proxy
                # Even if score didn't improve, update best_proxy if score is same?
                # Actually, if score is same, higher proxy is better for future SA moves.
                # But we track best_pts based on score.
            
        T *= alpha
        if T < T_min:
            T = T_min
            
    # Final check
    final_score = _score(best_pts)
    if final_score < best_score:
        best_score = final_score
        
    return best_pts

if __name__ == "__main__":
    pts = construct()
    print("Score:", _score(pts))

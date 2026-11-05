import time
import random
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    """
    Find a large subset of [1, 128] with no three distinct elements whose
    three pairwise gcds are all equal. Maximize cardinality.
    
    Strategy (STRUCTURAL CHANGE): Model as a 3-uniform hypergraph independent set problem.
    Use a simulated-annealing-style local search over the full 128-vertex space with
    multi-vertex removal moves (remove k, add up to k+1) to escape the local optima
    that trap single-removal 2-for-1 exchanges. Combine with a greedy seeding phase
    and a final exhaustive polish.
    """
    N = 128
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    # Precompute gcd table (1-indexed)
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g
    
    # Precompute all bad triples as a set of frozensets for O(1) lookup
    # Also build for each vertex a set of (other1, other2) pairs that form bad triples with it
    bad_triple_set = set()
    conflict_pairs = {i: set() for i in range(1, N + 1)}  # conflict_pairs[x] = set of (a,b) where {x,a,b} is bad
    degree = [0] * (N + 1)
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gt[a][b]
            for c in range(b + 1, N + 1):
                if gt[a][c] == g_ab and gt[b][c] == g_ab:
                    bad_triple_set.add((a, b, c))
                    conflict_pairs[a].add((b, c))
                    conflict_pairs[b].add((a, c))
                    conflict_pairs[c].add((a, b))
                    degree[a] += 1
                    degree[b] += 1
                    degree[c] += 1
    
    def is_valid_set(sel):
        """Check if sel is a valid independent set."""
        for a in sel:
            for b in sel:
                if b > a:
                    g_ab = gt[a][b]
                    for c in sel:
                        if c > b and gt[a][c] == g_ab and gt[b][c] == g_ab:
                            return False
        return True
    
    def can_add(x, sel):
        """True if x can be added to sel without creating a bad triple."""
        for (a, b) in conflict_pairs[x]:
            if a in sel and b in sel:
                return False
        return True
    
    def greedy_seed(order):
        """Build a valid set by adding elements in the given order."""
        sel = set()
        for x in order:
            if can_add(x, sel):
                sel.add(x)
        return sel
    
    def polish(sel):
        """Repeatedly add any excluded element that fits."""
        sel = set(sel)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in sel and can_add(x, sel):
                    sel.add(x)
                    changed = True
        return sel
    
    def multi_remove_search(sel, max_iters=300, max_remove=3):
        """
        Local search with multi-vertex removal: remove up to max_remove vertices,
        then greedily add back as many as possible. This can escape local optima
        that single-removal moves cannot.
        """
        sel = set(sel)
        best_size = len(sel)
        best_sel = set(sel)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            excluded = [x for x in range(1, N + 1) if x not in sel]
            
            # Try pure adds first
            for x in excluded:
                if can_add(x, sel):
                    sel.add(x)
                    improved = True
                    break
            if improved:
                if len(sel) > best_size:
                    best_size = len(sel)
                    best_sel = set(sel)
                continue
            
            # Try removing k vertices (k=1,2,...,max_remove) and refilling
            best_move = None
            best_gain = 0
            best_new_sel = None
            
            sel_list = list(sel)
            
            for k in range(1, min(max_remove, len(sel_list)) + 1):
                # Try random subsets of size k to remove
                num_tries = 20 if k <= 2 else 10
                for _ in range(num_tries):
                    to_remove = set(random.sample(sel_list, k))
                    temp = sel - to_remove
                    # Greedily add back
                    added = []
                    for x in excluded:
                        if can_add(x, temp | set(added)):
                            added.append(x)
                    
                    new_sel = temp | set(added)
                    new_size = len(new_sel)
                    gain = new_size - len(sel)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (to_remove, added)
                        best_new_sel = new_sel
            
            if best_move is not None and best_gain > 0:
                to_remove, added = best_move
                sel = best_new_sel
                improved = True
            elif best_move is not None and best_gain == 0:
                # Same size: accept with probability to explore
                if random.random() < 0.2:
                    to_remove, added = best_move
                    sel = best_new_sel
                    improved = True
            
            if not improved:
                # Try a larger random perturbation
                if random.random() < 0.3:
                    num_remove = random.randint(2, min(5, len(sel)))
                    to_remove = set(random.sample(sel_list, num_remove))
                    temp = sel - to_remove
                    added = []
                    for x in excluded:
                        if can_add(x, temp | set(added)):
                            added.append(x)
                    sel = temp | set(added)
                    improved = True
            
            if len(sel) > best_size:
                best_size = len(sel)
                best_sel = set(sel)
            
            if not improved:
                break
        
        return best_sel
    
    def simulated_annealing_search(sel, max_iters=500):
        """
        Simulated annealing: accept moves that decrease size with probability
        exp(-delta/T), allowing escape from local optima.
        """
        sel = set(sel)
        best_size = len(sel)
        best_sel = set(sel)
        T = 2.0
        cooling = 0.995
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            excluded = [x for x in range(1, N + 1) if x not in sel]
            sel_list = list(sel)
            
            # Generate a random move: remove k, add up to k+1
            k = random.randint(1, min(3, len(sel_list)))
            to_remove = set(random.sample(sel_list, k))
            temp = sel - to_remove
            
            # Greedily add back
            added = []
            for x in excluded:
                if can_add(x, temp | set(added)):
                    added.append(x)
            
            new_sel = temp | set(added)
            new_size = len(new_sel)
            delta = len(sel) - new_size  # positive means worse
            
            if delta <= 0 or random.random() < pow(2.718281828, -delta / T):
                sel = new_sel
                if len(sel) > best_size:
                    best_size = len(sel)
                    best_sel = set(sel)
            
            T *= cooling
        
        return best_sel
    
    # Phase 1: Multiple greedy seeds with different orderings
    best_set = set()
    best_size = 0
    
    # Seed 1: by increasing degree
    order1 = sorted(range(1, N + 1), key=lambda x: degree[x])
    s1 = greedy_seed(order1)
    s1 = polish(s1)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
    
    # Seed 2: by decreasing degree
    order2 = sorted(range(1, N + 1), key=lambda x: -degree[x])
    s2 = greedy_seed(order2)
    s2 = polish(s2)
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2
    
    # Seed 3: random order
    rng = random.Random(42)
    order3 = list(range(1, N + 1))
    rng.shuffle(order3)
    s3 = greedy_seed(order3)
    s3 = polish(s3)
    if len(s3) > best_size:
        best_size = len(s3)
        best_set = s3
    
    # Phase 2: Multi-vertex removal local search on best seed
    if time.time() < deadline - 10:
        s4 = multi_remove_search(best_set, max_iters=200, max_remove=3)
        s4 = polish(s4)
        if len(s4) > best_size:
            best_size = len(s4)
            best_set = s4
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 15:
        s5 = simulated_annealing_search(best_set, max_iters=400)
        s5 = polish(s5)
        if len(s5) > best_size:
            best_size = len(s5)
            best_set = s5
    
    # Phase 4: More randomized greedy seeds + local search
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        
        rng = random.Random(1000 + trial)
        scores = []
        for x in range(1, N + 1):
            score = degree[x] + rng.uniform(0, 10)
            scores.append((score, x))
        scores.sort()
        order = [x for _, x in scores]
        
        s = greedy_seed(order)
        s = polish(s)
        s = multi_remove_search(s, max_iters=80, max_remove=3)
        s = polish(s)
        
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Phase 5: Final aggressive local search on the best set
    if time.time() < deadline - 3:
        s_final = multi_remove_search(best_set, max_iters=300, max_remove=4)
        s_final = polish(s_final)
        if len(s_final) > best_size:
            best_size = len(s_final)
            best_set = s_final
    
    # Phase 6: One more SA pass if time allows
    if time.time() < deadline - 2:
        s_sa = simulated_annealing_search(best_set, max_iters=300)
        s_sa = polish(s_sa)
        if len(s_sa) > best_size:
            best_size = len(s_sa)
            best_set = s_sa
    
    if not best_set:
        best_set = {1}
    
    return sorted(best_set)
# EVOLVE-BLOCK-END

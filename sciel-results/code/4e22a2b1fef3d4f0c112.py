import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    # Precompute conflict graph using bitmasks
    # For each x, compute which pairs (b,c) with b<c, b,c in [1,n] have x | (b+c)
    # Two elements x,y conflict if there exist a,b,c in {x,y} forming a valid triple where the smallest divides the sum of the other two
    
    # Better approach: precompute for each pair (x,y) whether they can coexist
    # x and y conflict if:
    # - x < y and there exists z in [1,n] with y <= z such that x | (y+z) [x is 'a', y is 'b', z is 'c']
    #   OR x | (x+y) is false but we need to check triples
    # Actually, let me think more carefully.
    
    # Two elements x < y conflict if there exists a third element z (z != x, z != y) such that
    # one of x,y,z is the smallest and divides the sum of the other two.
    
    # But for the independent set approach, I need pairwise conflicts.
    # x and y are incompatible if there exists z such that {x,y,z} violates the property.
    
    # Let me precompute: for each x, what values of (b+c) mod x are forbidden?
    # If x is in the set, then for any b,c in the set with b > x, b <= c, we need x ∤ (b+c).
    
    # For pairwise incompatibility: x and y (x < y) are incompatible if:
    # Case A: x is smallest. Need z > x, z >= y (or z = y if y is one of b,c). 
    #   If z = y: need x ∤ (y+y) = 2y, i.e., x | 2y
    #   If z > y: need x ∤ (y+z) for all z in set with z >= y. But for pairwise, just check if such z could exist.
    #   Actually for pairwise conflict, we just need: does there exist ANY z such that {x,y,z} is invalid?
    #   If x | 2y, then z=y gives a violation (a=x, b=y, c=y).
    #   Also if x | (y+z) for some z > y, that's a potential conflict but z might not be in the set.
    #   
    # For a proper independent set, I should define conflicts as: x and y cannot both be in the set
    # because they would force a violation regardless of other elements.
    
    # The simplest pairwise conflict: x < y and x | (2*y) [i.e., a=x, b=y, c=y violates]
    # Also: x < y and x | (x + y) [i.e., a=x, b=x... wait, b must be > a, so this doesn't work]
    
    # Let me reconsider. The property is: for every a < b <= c in the set, a ∤ (b+c).
    # So if x < y are both in the set:
    # - If x is 'a' and y is 'b': need y <= c for some c in set, and x ∤ (y+c)
    # - If x is 'a' and y is 'c': need b in set with x < b <= y, and x ∤ (b+y)  
    # - If y is 'a': impossible since x < y
    
    # Pairwise incompatibility (must not both be in any valid set):
    # 1. x | (2y) -> triple (x, y, y) violates
    # 2. There exists b with x < b <= y such that x | (b + y) -> triple (x, b, y) violates
    #    This means: for some b in (x, y], x | (b+y), i.e., b ≡ -y (mod x)
    #    b = kx - y for some k, with x < b <= y
    #    x < kx - y <= y => x + y < kx <= 2y => (x+y)/x < k <= 2y/x
    #    k_min = floor((x+y)/x) + 1 = floor(y/x) + 1 (if (x+y)%x != 0) or (x+y)//x + 1
    #    Actually: k > (x+y)/x = 1 + y/x, so k_min = floor(y/x) + 1 (if y%x==0) or floor(y/x)+1
    #    k_min = (x + y + x - 1) // x = (2*x + y - 1) // x... let me just check all b in (x, y]
    
    # For efficiency, precompute pairwise conflicts
    conflict = [[False] * (n + 1) for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            # Check if x and y are incompatible
            # Condition 1: x | 2y
            if (2 * y) % x == 0:
                conflict[x][y] = True
                conflict[y][x] = True
                continue
            # Condition 2: exists b in (x, y] with x | (b + y)
            # b = k*x - y, need x < b <= y
            # x < k*x - y <= y
            # x + y < k*x <= 2*y
            # k > (x+y)/x, k <= 2y/x
            k_min = (x + y + x - 1) // x  # floor((x+y)/x) + 1 when not divisible
            # More precisely: k*x > x + y, so k > (x+y)/x
            # k_min = floor((x+y)/x) + 1
            k_min = (x + y) // x + 1
            if (x + y) % x == 0:
                k_min = (x + y) // x + 1
            else:
                k_min = (x + y) // x + 1
            k_max = (2 * y) // x
            found = False
            for k in range(k_min, k_max + 1):
                b = k * x - y
                if x < b <= y:
                    found = True
                    break
            if found:
                conflict[x][y] = True
                conflict[y][x] = True
    
    # Now solve maximum independent set on this graph using branch and bound
    # Use bitmasks for the candidate set
    
    def max_independent_set(nodes, time_limit):
        """Find maximum independent set using branch and bound."""
        best = set()
        best_size = 0
        start_t = time.time()
        
        # Greedy upper bound: for a set of nodes, upper bound on MIS
        def greedy_upper_bound(node_set):
            """Greedy coloring-based upper bound."""
            if not node_set:
                return 0
            # Simple bound: sort by degree, greedy coloring
            nodes_list = sorted(node_set, key=lambda x: sum(1 for y in node_set if x != y and conflict[x][y]))
            colors = {}
            max_color = 0
            for node in nodes_list:
                used_colors = set()
                for neighbor in node_set:
                    if neighbor != node and conflict[node][neighbor] and neighbor in colors:
                        used_colors.add(colors[neighbor])
                color = 1
                while color in used_colors:
                    color += 1
                colors[node] = color
                max_color = max(max_color, color)
            return max_color
        
        # Branch and bound
        def bnb(remaining, current_set):
            nonlocal best, best_size
            if time.time() > time_limit:
                return
            if not remaining:
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best = set(current_set)
                return
            
            # Upper bound
            ub = len(current_set) + greedy_upper_bound(remaining)
            if ub <= best_size:
                return
            
            # Pick a node to branch on
            # Choose node with highest degree in remaining
            node = max(remaining, key=lambda x: sum(1 for y in remaining if x != y and conflict[x][y]))
            
            # Branch 1: include node
            new_remaining = set()
            for v in remaining:
                if v != node and not conflict[node][v]:
                    new_remaining.add(v)
            bnb(new_remaining, current_set | {node})
            
            # Branch 2: exclude node
            new_remaining = set(remaining)
            new_remaining.discard(node)
            bnb(new_remaining, current_set)
        
        all_nodes = set(range(1, n + 1))
        bnb(all_nodes, set())
        return best
    
    # Strategy 1: Descending greedy (prioritize large numbers)
    def is_valid_addition(candidate, S):
        """Check if adding candidate to S maintains the property."""
        for b in S:
            if b <= candidate:
                continue
            k_min = (2 * b + candidate - 1) // candidate
            k_max = (n + b) // candidate
            for k in range(k_min, k_max + 1):
                c = k * candidate - b
                if c >= b and c in S:
                    return False
        
        for a in S:
            if a >= candidate:
                continue
            k_min = (2 * candidate + a - 1) // a
            k_max = (n + candidate) // a
            for k in range(k_min, k_max + 1):
                c = k * a - candidate
                if c >= candidate and c in S:
                    return False
        
        for a in S:
            if a >= candidate:
                continue
            k_min = (candidate // a) + 2
            k_max = (2 * candidate) // a
            for k in range(k_min, k_max + 1):
                b = k * a - candidate
                if a < b <= candidate and b in S:
                    return False
        
        return True
    
    def greedy(order, S_initial=None):
        """Greedy construction with given order."""
        S = set(S_initial) if S_initial else set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and is_valid_addition(x, S):
                S.add(x)
        return S
    
    def local_improve(S, max_iters=5000):
        """Try to improve S by swapping or adding elements."""
        S = set(S)
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline:
            improved = False
            iterations += 1
            
            for x in range(1, n + 1):
                if x in S:
                    continue
                if is_valid_addition(x, S):
                    S.add(x)
                    improved = True
                    break
            
            if not improved:
                S_list = sorted(S)
                for i, remove_x in enumerate(S_list):
                    if time.time() > deadline:
                        return S
                    S_new = S - {remove_x}
                    added_count = 0
                    for add_x in range(1, n + 1):
                        if add_x in S_new or add_x == remove_x:
                            continue
                        if is_valid_addition(add_x, S_new):
                            S_new.add(add_x)
                            added_count += 1
                            if added_count >= 2:
                                break
                    if added_count >= 2 and len(S_new) > len(S):
                        S = S_new
                        improved = True
                        break
        
        return S
    
    # Strategy 1: Descending greedy
    S1 = greedy(range(n, 0, -1))
    best_set = S1
    
    # Strategy 2: Ascending greedy
    if time.time() < deadline:
        S2 = greedy(range(1, n + 1))
        if len(S2) > len(best_set):
            best_set = S2
    
    # Strategy 3: Random shuffles
    random.seed(123)
    for seed in range(30):
        if time.time() > deadline:
            break
        random.seed(seed * 7919 + 13)
        order = list(range(1, n + 1))
        random.shuffle(order)
        S_rand = greedy(order)
        if len(S_rand) > len(best_set):
            best_set = S_rand
    
    # Strategy 4: Conflict-score ordering
    if time.time() < deadline:
        conflict_score = [0] * (n + 1)
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                for c in range(b, n + 1):
                    if (b + c) % a == 0:
                        conflict_score[a] += 1
        order_by_conflict = sorted(range(1, n + 1), key=lambda x: conflict_score[x])
        S_conflict = greedy(order_by_conflict)
        if len(S_conflict) > len(best_set):
            best_set = S_conflict
    
    # Strategy 5: Local improvement
    if time.time() < deadline:
        best_set = local_improve(best_set)
    
    # Strategy 6: Branch and bound on conflict graph (with time limit)
    if time.time() < deadline:
        remaining_time = deadline - time.time()
        if remaining_time > 5:  # Only try if we have enough time
            bnb_set = max_independent_set(set(range(1, n + 1)), time.time() + min(remaining_time - 2, 20))
            if len(bnb_set) > len(best_set):
                best_set = bnb_set
    
    # Strategy 7: Final local improvement
    if time.time() < deadline:
        best_set = local_improve(best_set)
    
    if not best_set:
        return [256]
    
    return sorted(best_set)

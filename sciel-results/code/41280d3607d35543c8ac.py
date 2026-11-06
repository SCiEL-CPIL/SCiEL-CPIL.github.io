import time
import random
import sys

N = 9
MAX_V = 1 << N

def construct():
    deadline = time.time() + 115.0
    sys.setrecursionlimit(10000)
    
    # Precompute neighbors
    neighbors = [[] for _ in range(MAX_V)]
    for v in range(MAX_V):
        for b in range(N):
            neighbors[v].append(v ^ (1 << b))
    
    # Precompute distance >= 2 check using popcount
    popcount = [0] * MAX_V
    for i in range(1, MAX_V):
        popcount[i] = popcount[i >> 1] + (i & 1)
    
    def dist_geq2(a, b):
        return popcount[a ^ b] >= 2
    
    best_path = []
    best_len = 0
    
    def is_valid_insertion(path, pos, v):
        """Check if vertex v can be inserted at position pos in path."""
        used = set(path)
        if v in used:
            return False
        # Check adjacency with neighbors in path
        if pos > 0 and not dist_geq2(v, path[pos - 1]):
            return False
        if pos < len(path) and not dist_geq2(v, path[pos]):
            return False
        # v must be adjacent to exactly one of the path vertices (the one it connects to)
        # Actually for insertion, v replaces an edge: path[pos-1]--path[pos] becomes path[pos-1]--v--path[pos]
        # So v must be adjacent to path[pos-1] AND path[pos]
        # Wait, that's not right. Let me reconsider.
        # Insertion at pos means: ... path[pos-1], v, path[pos], ...
        # v must be adjacent to path[pos-1] and path[pos]
        # And v must be distance >= 2 from all other path vertices
        if pos > 0:
            if popcount[v ^ path[pos - 1]] != 1:
                return False
        if pos < len(path):
            if popcount[v ^ path[pos]] != 1:
                return False
        for i, u in enumerate(path):
            if i == pos - 1 or i == pos:
                continue
            if not dist_geq2(v, u):
                return False
        return True
    
    def local_search_insert(path):
        """Try to insert unused vertices into the path."""
        if time.time() > deadline:
            return path
        
        new_path = path[:]
        used = set(new_path)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for pos in range(len(new_path) + 1):
                if time.time() > deadline:
                    break
                # Try to find a vertex that can be inserted at pos
                for v in range(MAX_V):
                    if v in used:
                        continue
                    if is_valid_insertion(new_path, pos, v):
                        new_path.insert(pos, v)
                        used.add(v)
                        improved = True
                        break
        return new_path
    
    def dfs_extend(path, used_set, depth_limit=100):
        """
        Systematic DFS with aggressive pruning.
        Returns the longest path found.
        """
        nonlocal best_path, best_len
        
        if time.time() > deadline:
            return path, used_set
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
        
        if len(path) >= depth_limit:
            return path, used_set
        
        cur = path[-1]
        
        # Get valid candidates: neighbors of cur not in used_set and distance >= 2 from all other used vertices
        candidates = []
        for nb in neighbors[cur]:
            if nb in used_set:
                continue
            # Check compatibility with all other used vertices
            compatible = True
            for u in used_set:
                if u == cur:
                    continue
                if popcount[nb ^ u] < 2:
                    compatible = False
                    break
            if compatible:
                candidates.append(nb)
        
        if not candidates:
            return path, used_set
        
        # Score by forward degree (Warnsdorff)
        def forward_deg(nb):
            new_used = used_set | {nb}
            deg = 0
            for nn in neighbors[nb]:
                if nn in new_used:
                    continue
                compat = True
                for u in new_used:
                    if u == nb:
                        continue
                    if popcount[nn ^ u] < 2:
                        compat = False
                        break
                if compat:
                    deg += 1
            return deg
        
        scored = [(forward_deg(nb), nb) for nb in candidates]
        scored.sort(key=lambda x: x[0])
        
        for deg, nb in scored:
            if time.time() > deadline:
                return path, used_set
            
            path.append(nb)
            used_set.add(nb)
            
            path, used_set = dfs_extend(path, used_set, depth_limit)
            
            path.pop()
            used_set.discard(nb)
        
        return path, used_set
    
    def greedy_extend(path, used_set, randomize=True):
        """Greedily extend using Warnsdorff's rule."""
        while True:
            if time.time() > deadline:
                return path, used_set
            
            cur = path[-1]
            
            # Get valid candidates
            candidates = []
            for nb in neighbors[cur]:
                if nb in used_set:
                    continue
                compatible = True
                for u in used_set:
                    if u == cur:
                        continue
                    if popcount[nb ^ u] < 2:
                        compatible = False
                        break
                if compatible:
                    candidates.append(nb)
            
            if not candidates:
                return path, used_set
            
            # Score by forward degree
            def forward_deg(nb):
                new_used = used_set | {nb}
                deg = 0
                for nn in neighbors[nb]:
                    if nn in new_used:
                        continue
                    compat = True
                    for u in new_used:
                        if u == nb:
                            continue
                        if popcount[nn ^ u] < 2:
                            compat = False
                            break
                    if compat:
                        deg += 1
                return deg
            
            scored = [(forward_deg(nb), nb) for nb in candidates]
            scored.sort(key=lambda x: x[0])
            
            if randomize and len(scored) > 1:
                k = min(3, len(scored))
                chosen = random.choice(scored[:k])[1]
            else:
                chosen = scored[0][1]
            
            path.append(chosen)
            used_set.add(chosen)
    
    def extend_both_ends(path, used_set):
        """Try to extend the path at both ends repeatedly."""
        new_path = path[:]
        new_used = set(used_set)
        
        # Prepend
        while True:
            if time.time() > deadline:
                break
            found = False
            first = new_path[0]
            for nb in neighbors[first]:
                if nb in new_used:
                    continue
                compatible = True
                for u in new_used:
                    if u == first:
                        continue
                    if popcount[nb ^ u] < 2:
                        compatible = False
                        break
                if compatible:
                    new_path.insert(0, nb)
                    new_used.add(nb)
                    found = True
                    break
            if not found:
                break
        
        # Append
        while True:
            if time.time() > deadline:
                break
            found = False
            last = new_path[-1]
            for nb in neighbors[last]:
                if nb in new_used:
                    continue
                compatible = True
                for u in new_used:
                    if u == last:
                        continue
                    if popcount[nb ^ u] < 2:
                        compatible = False
                        break
                if compatible:
                    new_path.append(nb)
                    new_used.add(nb)
                    found = True
                    break
            if not found:
                break
        
        return new_path, new_used
    
    # Main search
    random.seed(42)
    
    # Phase 1: Multiple greedy restarts to find a good starting point
    starts = list(range(MAX_V))
    random.shuffle(starts)
    
    for start in starts[:100]:
        if time.time() > deadline - 30:
            break
        
        path = [start]
        used_set = {start}
        path, used_set = greedy_extend(path, used_set, randomize=True)
        path, used_set = extend_both_ends(path, used_set)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 2: Systematic DFS from the best starting configuration
    if best_path:
        start = best_path[0]
        path = [start]
        used_set = {start}
        path, used_set = dfs_extend(path, used_set, depth_limit=150)
        path, used_set = extend_both_ends(path, used_set)
        path = local_search_insert(path)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 3: More DFS from different starts with shorter depth limits
    for start in starts:
        if time.time() > deadline - 5:
            break
        if best_path and start == best_path[0]:
            continue
        
        path = [start]
        used_set = {start}
        path, used_set = dfs_extend(path, used_set, depth_limit=120)
        path, used_set = extend_both_ends(path, used_set)
        path = local_search_insert(path)
        
        if len(path) > best_len:
            best_len = len(path)
            best_path = path[:]
    
    # Phase 4: Local search on the best path found
    if best_path:
        best_path = local_search_insert(best_path)
        best_len = len(best_path)
    
    return best_path if best_path else [0]

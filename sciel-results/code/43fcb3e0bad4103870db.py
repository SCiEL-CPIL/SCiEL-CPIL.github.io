import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n = 20
    all_mask = (1 << n) - 1
    
    def common_count(mask, i, j, k):
        return (mask[i] & mask[j] & mask[k]).bit_count()
    
    def is_valid(mask):
        for i in range(n):
            mi = mask[i]
            for j in range(i + 1, n):
                mij = mi & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        return False
        return True
    
    def edge_count(mask):
        return sum(mask[i].bit_count() for i in range(n)) // 2
    
    def find_violations(mask):
        viols = []
        for i in range(n):
            mi = mask[i]
            for j in range(i + 1, n):
                mij = mi & mask[j]
                for k in range(j + 1, n):
                    if (mij & mask[k]).bit_count() >= 3:
                        viols.append((i, j, k))
        return viols

    def check_add(mask, u, v):
        # Check if adding edge (u, v) creates a violation
        mu = mask[u]
        mv = mask[v]
        
        # Add edge
        mask[u] = mu | (1 << v)
        mask[v] = mv | (1 << u)
        
        valid = True
        # Check triples involving u or v
        for w in range(n):
            if w == u or w == v:
                continue
            mw = mask[w]
            # Triple {u, v, w}
            if (mask[u] & mask[v] & mw).bit_count() >= 3:
                valid = False
                break
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                mz = mask[z]
                # Triple {u, w, z}
                if (mask[u] & mw & mz).bit_count() >= 3:
                    valid = False
                    break
            if not valid:
                break
            # Triple {v, w, z}
            for z in range(w + 1, n):
                if z == u or z == v:
                    continue
                mz = mask[z]
                if (mask[v] & mw & mz).bit_count() >= 3:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Revert
            mask[u] = mu
            mask[v] = mv
            return False
        
        return True

    def repair(mask):
        max_iter = 10000
        for _ in range(max_iter):
            if time.time() > deadline:
                return False
            viols = find_violations(mask)
            if not viols:
                return True
            
            i, j, k = viols[0]
            # Remove an edge incident to one of the vertices in the violation
            candidates = []
            for v in range(n):
                if v != i and mask[i] & (1 << v):
                    candidates.append((i, v))
                if v != j and mask[j] & (1 << v):
                    candidates.append((j, v))
                if v != k and mask[k] & (1 << v):
                    candidates.append((k, v))
            
            if not candidates:
                return False
            
            u, v = random.choice(candidates)
            mask[u] &= ~(1 << v)
            mask[v] &= ~(1 << u)
        
        return not find_violations(mask)

    def add_edges(mask):
        improved = True
        while improved:
            if time.time() > deadline:
                return
            improved = False
            # Shuffle edges to avoid bias
            edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not (mask[i] & (1 << j))]
            random.shuffle(edges)
            for u, v in edges:
                if time.time() > deadline:
                    return
                if check_add(mask, u, v):
                    improved = True

    def local_search_swap(mask):
        """Try to improve by swapping edges: remove (a,b), add (c,d)"""
        if time.time() > deadline:
            return False
        current_count = edge_count(mask)
        
        # Get all edges
        edges = []
        non_edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i] & (1 << j):
                    edges.append((i, j))
                else:
                    non_edges.append((i, j))
        
        if not edges or not non_edges:
            return False
            
        random.shuffle(edges)
        random.shuffle(non_edges)
        
        # Try a few random swaps
        for _ in range(50):
            if time.time() > deadline:
                return False
            if not edges:
                break
            a, b = random.choice(edges)
            c, d = random.choice(non_edges)
            
            # Check if removing (a,b) and adding (c,d) keeps validity
            # First remove (a,b)
            mask[a] &= ~(1 << b)
            mask[b] &= ~(1 << a)
            
            # Try to add (c,d)
            if check_add(mask, c, d):
                # Success - we've swapped
                edges.remove((a, b))
                non_edges.remove((c, d))
                # Update counts implicitly
                return True
            else:
                # Revert the removal
                mask[a] |= (1 << b)
                mask[b] |= (1 << a)
        
        return False

    best_mask = None
    best_count = -1
    random.seed(42)
    
    # Try various densities
    densities = [0.4, 0.5, 0.6, 0.3, 0.7, 0.45, 0.55, 0.65, 0.5, 0.4]
    
    for p in densities:
        if time.time() > deadline:
            break
        
        # Multiple restarts per density
        for restart in range(15):
            if time.time() > deadline:
                break
            
            mask = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        mask[i] |= (1 << j)
                        mask[j] |= (1 << i)
            
            if repair(mask):
                add_edges(mask)
                
                # Try local search swaps to potentially improve further
                swap_improved = True
                while swap_improved and time.time() < deadline:
                    swap_improved = local_search_swap(mask)
                
                ec = edge_count(mask)
                if ec > best_count:
                    best_count = ec
                    best_mask = mask[:]

    if best_mask is None:
        # Fallback: empty graph or sparse
        return []
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_mask[i] & (1 << j):
                edges.append([i, j])
    
    return edges

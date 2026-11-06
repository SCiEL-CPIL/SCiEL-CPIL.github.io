"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def _has_k222(adj, n):
    """Check if graph with adjacency bitmasks adj on n vertices contains K2,2,2 as a subgraph."""
    # K2,2,2: parts {a,b}, {c,d}, {e,f} with all 12 cross-edges
    # For each edge (a,b), common neighbors CN must contain a C4
    # C4 exists iff two vertices in CN share >=2 common neighbors in CN
    for a in range(n):
        for b in range(a + 1, n):
            if not ((adj[a] >> b) & 1):
                continue
            cn_mask = adj[a] & adj[b]
            cn_mask &= ~(1 << a)
            cn_mask &= ~(1 << b)
            cnt = cn_mask.bit_count()
            if cnt < 4:
                continue
            # Early exit: if any vertex in CN has degree >= 3 within CN, 
            # we might have C4 but need to check properly
            # Get list of common neighbors
            cn_list = []
            m = cn_mask
            while m:
                lsb = m & (-m)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                m ^= lsb
            L = len(cn_list)
            # Check pairs in CN for >=2 common neighbors within CN
            for i in range(L):
                u = cn_list[i]
                for j in range(i + 1, L):
                    v = cn_list[j]
                    common_uv = adj[u] & adj[v] & cn_mask
                    if common_uv.bit_count() >= 2:
                        return True
    return False

def _is_valid(adj, n):
    """Check both color classes are K2,2,2-free."""
    if _has_k222(adj, n):
        return False
    comp = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        comp[i] = full ^ adj[i] ^ (1 << i)
    if _has_k222(comp, n):
        return False
    return True

def _build_paley(q):
    """Build Paley graph on q vertices (q prime, q ≡ 1 mod 4)."""
    squares = set()
    for x in range(1, q):
        squares.add(x * x % q)
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            diff = (j - i) % q
            if diff in squares:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def _try_extend(adj, n, max_n, deadline, rng):
    """Try to extend a valid coloring by adding vertices greedily with randomization."""
    best_adj = adj[:]
    best_n = n
    for target_n in range(n + 1, max_n + 1):
        if time.time() > deadline:
            break
        new_adj = best_adj + [0] * (target_n - n)
        for attempt in range(50):
            if time.time() > deadline:
                break
            candidate = best_adj + [0] * (target_n - n)
            success = True
            for new_v in range(n, target_n):
                # Assign colors to edges from new_v to existing vertices
                found = False
                for _ in range(100):
                    if time.time() > deadline:
                        return best_adj, best_n
                    # Randomly assign colors
                    for old_v in range(new_v):
                        if rng.random() < 0.5:
                            candidate[new_v] |= (1 << old_v)
                            candidate[old_v] |= (1 << new_v)
                        else:
                            candidate[new_v] &= ~(1 << old_v)
                            candidate[old_v] &= ~(1 << new_v)
                    if _is_valid(candidate, target_n):
                        found = True
                        break
                if not found:
                    success = False
                    break
            if success:
                best_adj = candidate
                best_n = target_n
                break
    return best_adj, best_n

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    rng = random.Random(42)
    
    # Try Paley graphs for primes ≡ 1 mod 4
    paley_primes = [29, 17, 13, 5]
    best_adj = None
    best_n = 0
    
    for q in paley_primes:
        if time.time() > deadline:
            break
        adj = _build_paley(q)
        if _is_valid(adj, q):
            best_adj = adj
            best_n = q
            break
    
    if best_n == 0:
        # Fallback: use Paley(5) which should always work
        best_adj = _build_paley(5)
        best_n = 5
    
    # Try to extend from best valid coloring
    if best_n < 31:
        extended_adj, extended_n = _try_extend(best_adj, best_n, 31, deadline, rng)
        if extended_n > best_n:
            best_adj = extended_adj
            best_n = extended_n
    
    # Build colors list
    colors = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                colors.append(0)
            else:
                colors.append(1)
    return {'n': best_n, 'colors': colors}
# EVOLVE-BLOCK-END

import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    n = 512
    nodes = list(range(1, n + 1))
    
    # 1. Build adjacency bitsets
    adj = [0] * (n + 1)
    
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            g = gcd(i, j)
            l = (i // g) * j
            if l <= n:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))
        
        if time.time() > deadline:
            return [1, 2]
    
    # 2. Greedy removal to find a large initial clique
    # Start with all nodes, iteratively remove one from each conflicting pair
    # preferring to remove the node with fewer compatible neighbors
    def greedy_initial_clique():
        if time.time() > deadline:
            return list(range(1, 23))
        
        # Build degree (compatibility count) for each node
        compat_count = [0] * (n + 1)
        for i in range(1, n + 1):
            compat_count[i] = adj[i].bit_count()
        
        # Start with all nodes
        in_set = [False] * (n + 1)
        for i in range(1, n + 1):
            in_set[i] = True
        
        # Iteratively remove nodes from conflicting pairs
        # Process pairs in order of lcm (worst conflicts first)
        removed = [False] * (n + 1)
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Find a conflicting pair where both are still in set
            # Prioritize pairs where one has low compatibility
            best_i = -1
            best_j = -1
            best_score = -1
            for i in range(1, n + 1):
                if not in_set[i]:
                    continue
                # Check against all j > i that are in set and conflict with i
                # Use bit operations: find nodes in set that conflict with i
                # conflict_mask[i] = nodes NOT adjacent to i (excluding i itself)
                # Actually adj[i] has neighbors; non-neighbors among 1..n are ~(adj[i]) & ((1<<n)-1)
                # Let's just iterate
                for j in range(i + 1, n + 1):
                    if not in_set[j]:
                        continue
                    if not (adj[i] >> (j - 1)) & 1:
                        # Conflicting pair
                        # Score: prefer removing the one with fewer compat neighbors
                        score = min(compat_count[i], compat_count[j])
                        if score > best_score:
                            best_score = score
                            best_i = i
                            best_j = j
            if best_i == -1:
                break
            # Remove the one with fewer compat neighbors
            if compat_count[best_i] <= compat_count[best_j]:
                in_set[best_i] = False
                # Update compat counts for nodes compatible with best_i
                # Actually just track in_set
            else:
                in_set[best_j] = False
            changed = True
        
        # Extract clique
        clique = [i for i in range(1, n + 1) if in_set[i]]
        return clique
    
    greedy_clique = greedy_initial_clique()
    best_clique = greedy_clique
    best_clique_size = len(best_clique)

    # 3. Also try divisor-based seeds and take the best
    def get_divisors(L):
        divs = []
        for k in range(1, int(L**0.5) + 1):
            if L % k == 0:
                divs.append(k)
                if k != L // k:
                    divs.append(L // k)
        return divs
    
    best_div_count = 0
    best_L = 1
    for L in range(1, n + 1):
        d = get_divisors(L)
        if len(d) > best_div_count:
            best_div_count = len(d)
            best_L = L
    
    div_clique = get_divisors(best_L)
    if len(div_clique) > best_clique_size:
        best_clique = div_clique
        best_clique_size = len(div_clique)
    
    # 4. Bron-Kerbosch with bitsets and improved pivot
    # Only search for cliques larger than best_clique_size
    # Start from the best clique found so far and try to extend it
    def bron_kerbosch(R, P, X):
        nonlocal best_clique, best_clique_size
        
        if time.time() > deadline:
            return
            
        # Pruning: can't beat best
        if len(R) + P.bit_count() <= best_clique_size:
            return

        if P == 0:
            if X == 0:
                if len(R) > best_clique_size:
                    best_clique_size = len(R)
                    best_clique = list(R)
            return

        # Pivot selection: choose u in P | X that maximizes (P & adj[u]).bit_count()
        # Sample a subset if P|X is large for speed
        pivot = -1
        max_count = -1
        
        px = P | X
        # If too many bits, sample
        px_count = px.bit_count()
        if px_count > 200:
            # Sample 100 random-ish bits
            px_sample = 0
            temp = px
            step = max(1, px_count // 100)
            cnt = 0
            while temp and cnt < 100:
                lsb = temp & -temp
                if cnt % step == 0:
                    px_sample |= lsb
                temp ^= lsb
                cnt += 1
            px = px_sample
        
        while px:
            lsb = px & -px
            u = lsb.bit_length()
            count = (P & adj[u]).bit_count()
            if count > max_count:
                max_count = count
                pivot = u
            px ^= lsb
            
        if pivot == -1:
            return

        # Candidates: P & ~adj[pivot]
        candidates = P & ~adj[pivot]
        
        while candidates:
            lsb = candidates & -candidates
            v = lsb.bit_length()
            
            # Recurse
            new_P = P & adj[v]
            new_X = X & adj[v]
            
            R.append(v)
            bron_kerbosch(R, new_P, new_X)
            R.pop()
            
            # Update P and X
            P ^= lsb
            X |= lsb
            candidates ^= lsb
            
            if time.time() > deadline:
                return

    # Start Bron-Kerbosch from the best clique found
    # P = nodes compatible with all in best_clique, not yet in R
    # X = empty (we're building from scratch but with a good lower bound)
    # Actually, let's start fresh but with the good lower bound for pruning
    initial_P = (1 << n) - 1
    initial_X = 0
    initial_R = []
    
    bron_kerbosch(initial_R, initial_P, initial_X)
    
    # 5. Greedy improvement: try to add nodes to best_clique
    if time.time() < deadline - 2.0:
        improved = list(best_clique)
        improved_set = set(improved)
        changed = True
        while changed and time.time() < deadline - 1.0:
            changed = False
            for v in range(1, n + 1):
                if v in improved_set:
                    continue
                # Check if v can be added to improved using bitset
                # v is compatible with all in improved if adj[v] contains all bits of improved
                # Build mask for improved
                # Actually, check: for each u in improved, v must be adjacent to u
                # Use bitset: adj[v] should have bits set for all u in improved
                can_add = True
                for u in improved:
                    if not (adj[v] >> (u - 1)) & 1:
                        can_add = False
                        break
                if can_add:
                    improved.append(v)
                    improved_set.add(v)
                    changed = True
                    if len(improved) > best_clique_size:
                        best_clique_size = len(improved)
                        best_clique = improved
                    break  # Restart scan after adding
    
    return best_clique

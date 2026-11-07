import time
import numpy as np
import itertools

def _build_comparability_matrix():
    """Build a 4096x4096 boolean matrix where comp[i,j]=True iff subsets i,j are comparable."""
    comp = np.zeros((4096, 4096), dtype=bool)
    for i in range(4096):
        # Enumerate all proper subsets of i
        sub = (i - 1) & i
        while True:
            comp[i, sub] = True
            comp[sub, i] = True
            if sub == 0:
                break
            sub = (sub - 1) & i
        # Enumerate all proper supersets of i
        complement = (~i) & 0xFFF
        sub_comp = complement
        while True:
            j = i | sub_comp
            if j != i:
                comp[i, j] = True
                comp[j, i] = True
            if sub_comp == 0:
                break
            sub_comp = (sub_comp - 1) & complement
    return comp

def _count_comparable(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(sets)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if len(sets[i]) < len(sets[j]):
                if all(x in sets[j] for x in sets[i]):
                    count += 1
            elif len(sets[j]) < len(sets[i]):
                if all(x in sets[i] for x in sets[j]):
                    count += 1
    return count

def _construct_base():
    deadline = time.time() + 80
    
    comp = _build_comparability_matrix()
    
    # Initial solution: 40 nodes with highest degree
    deg = comp.sum(axis=1)
    top_nodes = np.argsort(deg)[::-1][:40]
    selected = set(int(x) for x in top_nodes)
    selected_list = list(selected)
    
    # Precompute mask for selected nodes
    # We will maintain an array 'in_set' of shape (4096,) boolean
    in_set = np.zeros(4096, dtype=bool)
    in_set[selected_list] = True
    
    # Current score
    sub_comp = comp[np.ix_(selected_list, selected_list)]
    current_score = int(sub_comp.sum() // 2)
    
    best_score = current_score
    best_selected = selected_list.copy()
    
    # Iterated Local Search
    while time.time() < deadline:
        improved = True
        while improved:
            improved = False
            for i in range(len(selected_list)):
                if time.time() > deadline:
                    break
                u = selected_list[i]
                # Remove u
                in_set[u] = False
                selected_list.pop(i)
                
                # Calculate gain for each candidate v not in set
                # gain[v] = sum(comp[v, selected_list]) - sum(comp[u, selected_list])
                # Note: comp[u, selected_list] is constant for this iteration
                # We want to maximize sum(comp[v, selected_list])
                
                # Vectorized calculation
                # candidates are those where in_set is False
                # We need sum of comp[v, selected_list] for all v
                
                # comp[:, selected_list] gives a (4096, 40) matrix
                # Sum along axis 1 gives the number of neighbors in selected_list for each node
                neighbor_counts = comp[:, selected_list].sum(axis=1)
                
                # Mask out nodes already in set
                neighbor_counts[in_set] = -1
                
                # Find best candidate
                best_v = int(np.argmax(neighbor_counts))
                
                # The gain is neighbor_counts[best_v] - (neighbors of u in current set)
                # Wait, we already removed u. So 'selected_list' is S \ {u}.
                # The term to subtract is the number of neighbors u had in S \ {u}.
                # Let's call it deg_u_in_S_minus_u.
                # We can calculate this before removing u, or just compare new score.
                
                # Simpler: just calculate new score if we add best_v
                # But we can estimate gain.
                # Let's just add best_v and check if score improved.
                
                selected_list.append(best_v)
                in_set[best_v] = True
                
                # Recalculate score
                sub_comp = comp[np.ix_(selected_list, selected_list)]
                new_score = int(sub_comp.sum() // 2)
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    # Update best if necessary
                    if current_score > best_score:
                        best_score = current_score
                        best_selected = selected_list.copy()
                else:
                    # Revert
                    selected_list.pop()
                    in_set[best_v] = False
                    selected_list.insert(i, u)
                    in_set[u] = True
        
        # Perturb
        if time.time() > deadline:
            break
        
        # Randomly swap 2 elements
        if len(selected_list) >= 2:
            idx1, idx2 = np.random.choice(len(selected_list), 2, replace=False)
            u1 = selected_list[idx1]
            u2 = selected_list[idx2]
            
            # Find 2 random nodes not in set
            # Create list of available nodes
            # This is slow if done naively. 
            # We can just pick random indices from 0..4095 and check if in_set.
            
            candidates = []
            while len(candidates) < 2:
                r = np.random.randint(0, 4096)
                if not in_set[r]:
                    candidates.append(r)
            
            v1, v2 = candidates
            
            # Apply swap
            in_set[u1] = False
            in_set[u2] = False
            selected_list[idx1] = v1
            selected_list[idx2] = v2
            in_set[v1] = True
            in_set[v2] = True
            
            # Update score
            sub_comp = comp[np.ix_(selected_list, selected_list)]
            current_score = int(sub_comp.sum() // 2)
            
            if current_score > best_score:
                best_score = current_score
                best_selected = selected_list.copy()

    # Convert best_selected to list of lists
    result = []
    for idx in best_selected:
        subset = []
        for bit in range(12):
            if (idx >> bit) & 1:
                subset.append(bit)
        result.append(subset)
    
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    try:
        result = _construct_base()
        assert len(result) == 40
        seen = set()
        for s in result:
            assert isinstance(s, (list, tuple))
            for x in s:
                assert isinstance(x, int) and not isinstance(x, bool)
                assert 0 <= x <= 11
            fs = frozenset(s)
            assert fs not in seen
            seen.add(fs)
        return result
    except Exception:
        # Fallback
        chain = [list(range(i)) for i in range(13)]
        extra = []
        for r in range(1, 12):
            for start in range(12 - r + 1):
                s = list(range(start, start + r))
                if frozenset(s) not in set(frozenset(c) for c in chain) and frozenset(s) not in set(frozenset(e) for e in extra):
                    extra.append(s)
                if len(chain) + len(extra) >= 40:
                    break
            if len(chain) + len(extra) >= 40:
                break
        result = chain + extra[:40 - len(chain)]
        while len(result) < 40:
            for r in range(13):
                for combo in itertools.combinations(range(12), r):
                    s = list(combo)
                    if frozenset(s) not in set(frozenset(x) for x in result):
                        result.append(s)
                        break
                if len(result) >= 40:
                    break
        result = result[:40]
        return [list(s) for s in result]

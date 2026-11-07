import time
import itertools
import numpy as np

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
    deadline = time.time() + 85
    
    # Build comparability matrix
    comp = _build_comparability_matrix()
    
    # Degree of each node
    deg = comp.sum(axis=1)
    
    # Greedy initial solution
    selected = []
    used = np.zeros(4096, dtype=bool)
    
    # We want to maximize the number of edges in the induced subgraph.
    # Greedy: pick node with max degree to current selected set.
    
    # Initialize with a high degree node
    best_node = int(np.argmax(deg))
    selected.append(best_node)
    used[best_node] = True
    
    # Maintain count of neighbors in selected for each node
    # count[v] = sum(comp[v, u] for u in selected)
    count = np.zeros(4096, dtype=np.int32)
    count[best_node] = 0 # Self loop is 0
    
    # Add 39 more nodes
    for _ in range(39):
        # Find node with max count not used
        # Mask used nodes
        counts_masked = count.copy()
        counts_masked[used] = -1
        best_node = int(np.argmax(counts_masked))
        
        # If best count is negative, it means no unselected node is connected to selected?
        # This shouldn't happen unless selected is empty or graph is disconnected (it's not).
        # But wait, if selected is {v}, count[u] is 1 if u~v, 0 otherwise.
        # So we pick a neighbor.
        
        selected.append(best_node)
        used[best_node] = True
        
        # Update counts
        # For all neighbors of best_node, increment their count
        neighbors = np.where(comp[best_node])[0]
        count[neighbors] += 1
        
    current_score = 0
    sel_arr = np.array(selected)
    sub_comp = comp[np.ix_(sel_arr, sel_arr)]
    current_score = int(sub_comp.sum() // 2)
    
    # Local Search
    improved = True
    max_iter = 1000
    iter_count = 0
    
    while improved and iter_count < max_iter and time.time() < deadline:
        improved = False
        iter_count += 1
        
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            
            v_remove = selected[i]
            
            # Calculate loss of removing v_remove
            # Loss is number of neighbors of v_remove in selected
            neighbors_in_sel = np.sum(comp[v_remove, sel_arr])
            
            # Temporarily remove v_remove
            selected[i] = -1 # Placeholder
            used[v_remove] = False
            # Update counts
            neighbors = np.where(comp[v_remove])[0]
            count[neighbors] -= 1
            
            # Find best node to add
            counts_masked = count.copy()
            counts_masked[used] = -1
            # Also exclude v_remove itself (it's not used now, but we don't want to add it back immediately unless it's best)
            # Actually, if we add it back, score is same. We want a different node.
            # So set count[v_remove] = -1 as well to force a swap
            counts_masked[v_remove] = -1
            
            v_add = int(np.argmax(counts_masked))
            
            # Gain is count[v_add]
            gain = count[v_add]
            
            if gain > neighbors_in_sel:
                # Perform swap
                selected[i] = v_add
                used[v_add] = True
                
                # Update counts
                neighbors_add = np.where(comp[v_add])[0]
                count[neighbors_add] += 1
                
                # Recalculate score? Or update incrementally?
                # Old score - neighbors_in_sel + gain
                # But neighbors_in_sel was calculated with v_remove in sel?
                # comp[v_remove, sel_arr] includes v_remove? No, comp[i,i] is 0.
                # So neighbors_in_sel is correct.
                # However, the gain `count[v_add]` was calculated with v_remove REMOVED.
                # So the new score is current_score - neighbors_in_sel + gain.
                current_score = current_score - neighbors_in_sel + gain
                
                improved = True
            else:
                # Revert
                selected[i] = v_remove
                used[v_remove] = True
                neighbors = np.where(comp[v_remove])[0]
                count[neighbors] += 1
    
    # Convert to list of lists
    result = []
    for idx in selected:
        subset = []
        for bit in range(12):
            if (idx >> bit) & 1:
                subset.append(bit)
        result.append(subset)
    
    assert len(result) == 40
    assert len(set(frozenset(s) for s in result)) == 40
    
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

"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

def pebble(dag, fast_memory_budget):
    t_start = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    n = len(dag)
    if n == 0:
        return []
    
    vertices = []
    parents = []
    is_output = []
    
    for i, item in enumerate(dag):
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                v_id = item[0]
                v_parents = item[1]
                v_output = item[2] if len(item) > 2 else False
                vertices.append(v_id)
                parents.append(list(v_parents))
                is_output.append(bool(v_output))
            else:
                vertices.append(item[0])
                parents.append([])
                is_output.append(False)
        elif isinstance(item, dict):
            v_id = item.get('id', item.get('vertex', i))
            v_parents = item.get('parents', item.get('preds', []))
            v_output = item.get('output', item.get('is_output', False))
            vertices.append(v_id)
            parents.append(list(v_parents))
            is_output.append(bool(v_output))
        else:
            vertices.append(item)
            parents.append([])
            is_output.append(False)
    
    vid_to_idx = {}
    for i, v in enumerate(vertices):
        vid_to_idx[v] = i
    
    children = defaultdict(list)
    for i, p_list in enumerate(parents):
        for p in p_list:
            if p in vid_to_idx:
                children[vid_to_idx[p]].append(i)
    
    remaining_uses = [0] * n
    for i in range(n):
        remaining_uses[i] = len(children[i])
    
    fast_mem = set()
    operations = []
    
    blue = [False] * n
    for i in range(n):
        if len(parents[i]) == 0:
            blue[i] = True
    
    for i in range(n):
        if time.time() - t_start > DEADLINE:
            break
        
        for p in parents[i]:
            p_idx = vid_to_idx.get(p)
            if p_idx is not None and p_idx not in fast_mem:
                if blue[p_idx]:
                    operations.append(["load", vertices[p_idx]])
                    fast_mem.add(p_idx)
        
        operations.append(["compute", vertices[i]])
        fast_mem.add(i)
        
        must_store = is_output[i] or remaining_uses[i] == 0
        
        if must_store:
            operations.append(["store", vertices[i]])
            fast_mem.discard(i)
            blue[i] = True
        else:
            if len(fast_mem) > fast_memory_budget:
                best_evict = None
                best_score = float('inf')
                for j in fast_mem:
                    if j == i:
                        continue
                    score = remaining_uses[j]
                    if score < best_score:
                        best_score = score
                        best_evict = j
                if best_evict is not None:
                    operations.append(["store", vertices[best_evict]])
                    fast_mem.discard(best_evict)
                    blue[best_evict] = True
        
        for p in parents[i]:
            p_idx = vid_to_idx.get(p)
            if p_idx is not None:
                remaining_uses[p_idx] -= 1
    
    for i in range(n):
        if is_output[i] and i in fast_mem:
            operations.append(["store", vertices[i]])
            fast_mem.discard(i)
            blue[i] = True
    
    if len(operations) > 20000:
        operations = operations[:20000]
    
    return operations

# EVOLVE-BLOCK-END

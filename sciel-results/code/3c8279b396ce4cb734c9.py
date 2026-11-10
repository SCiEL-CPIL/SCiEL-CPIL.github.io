import itertools
import random

def construct(n):
    if n == 1:
        return "1"
    
    all_perms = set(itertools.permutations(range(1, n + 1)))
    visited_perms = set()
    
    random.seed(42)
    start_nodes = list(itertools.permutations(range(1, n + 1), n - 1))
    current_node = start_nodes[0]
    
    word = list(current_node)
    
    # Cache for neighbor info to avoid recalculating set membership etc if we revisit nodes?
    # We might revisit nodes.
    # But the "unvisited" status changes, so we can't cache the score.
    # We can cache the "is_perm" and "next_node" mapping.
    
    neighbor_struct_cache = {}
    
    def get_neighbor_structs(node):
        if node in neighbor_struct_cache:
            return neighbor_struct_cache[node]
            
        structs = []
        node_set = set(node)
        has_distinct = (len(node_set) == n - 1)
        
        for x in range(1, n + 1):
            new_node = node[1:] + (x,)
            is_perm = False
            perm_tuple = None
            if has_distinct and x not in node_set:
                is_perm = True
                perm_tuple = node + (x,)
            structs.append((new_node, x, is_perm, perm_tuple))
            
        neighbor_struct_cache[node] = structs
        return structs

    while len(visited_perms) < len(all_perms):
        structs = get_neighbor_structs(current_node)
        
        candidates = []
        
        for new_node, x, is_perm, perm_tuple in structs:
            # Calculate lookahead score: number of unvisited perm edges from new_node
            new_node_set = set(new_node)
            if len(new_node_set) == n - 1:
                # Find the unique x' not in new_node
                # Since new_node has n-1 distinct elements from 1..n, there is exactly one missing.
                # We can find it quickly.
                # Or just loop 1..n.
                unvisited_count = 0
                for xx in range(1, n + 1):
                    if xx not in new_node_set:
                        p = new_node + (xx,)
                        if p not in visited_perms:
                            unvisited_count = 1 # Can only be 1
                        break
            else:
                unvisited_count = 0
                
            if is_perm and perm_tuple not in visited_perms:
                score = 10000 + unvisited_count
            else:
                score = unvisited_count
                
            candidates.append((score, new_node, x, is_perm, perm_tuple))
            
        if not candidates:
            break
            
        # Sort by score descending
        # If tie, maybe prefer lower x for determinism?
        candidates.sort(key=lambda t: (-t[0], t[2]))
        
        best_score, new_node, x, is_perm, perm_tuple = candidates[0]
        
        word.append(x)
        if is_perm and perm_tuple not in visited_perms:
            visited_perms.add(perm_tuple)
            
        current_node = new_node
        
    return "".join(map(str, word))

<<<<<<< SEARCH
    def local_search(adj_mask, max_iters=300):
        """Local search with edge additions and swaps."""
        adj_mask = repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                      if not (adj_mask[i] & bit(j))]
            
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                    improved = True
                    break
            
            if improved:
                adj_mask = greedy_add(adj_mask)
                continue
            
            # Try swaps: remove one edge, add another
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                        if adj_mask[i] & bit(j)]
            
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            swapped = False
            for i in range(min(30, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove (r_u, r_v)
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                # Add (a_u, a_v)
                adj_mask[a_u] |= bit(a_v)
                adj_mask[a_v] |= bit(a_u)
                
                if find_violation(adj_mask) is None:
                    adj_mask = greedy_add(adj_mask)
                    swapped = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= bit(r_v)
                    adj_mask[r_v] |= bit(r_u)
                    adj_mask[a_u] &= ~bit(a_v)
                    adj_mask[a_v] &= ~bit(a_u)
            
            if not swapped:
                # Try removing a random edge to escape local optimum
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~bit(r_v)
                    adj_mask[r_v] &= ~bit(r_u)
                    adj_mask = greedy_add(adj_mask)
        
        return adj_mask
=======
    def local_search(adj_mask, max_iters=300):
        """Local search with edge additions and swaps."""
        adj_mask = repair(adj_mask)
        adj_mask = greedy_add(adj_mask)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to add missing edges
            missing = [(i, j) for i in range(n) for j in range(i + 1, n) 
                      if not (adj_mask[i] & bit(j))]
            
            if not missing:
                break
            
            random.shuffle(missing)
            improved = False
            
            for u, v in missing:
                if can_add(u, v, adj_mask):
                    adj_mask[u] |= bit(v)
                    adj_mask[v] |= bit(u)
                    improved = True
                    break
            
            if improved:
                adj_mask = greedy_add(adj_mask)
                continue
            
            # Try swaps: remove one edge, add another
            all_edges = [(i, j) for i in range(n) for j in range(i + 1, n) 
                        if adj_mask[i] & bit(j)]
            
            if not all_edges:
                break
            
            random.shuffle(all_edges)
            random.shuffle(missing)
            
            swapped = False
            for i in range(min(50, len(all_edges), len(missing))):
                if time.time() > deadline:
                    break
                
                r_u, r_v = all_edges[i]
                a_u, a_v = missing[i]
                
                # Remove (r_u, r_v)
                adj_mask[r_u] &= ~bit(r_v)
                adj_mask[r_v] &= ~bit(r_u)
                # Add (a_u, a_v)
                adj_mask[a_u] |= bit(a_v)
                adj_mask[a_v] |= bit(a_u)
                
                if find_violation(adj_mask) is None:
                    adj_mask = greedy_add(adj_mask)
                    swapped = True
                    break
                else:
                    # Revert
                    adj_mask[r_u] |= bit(r_v)
                    adj_mask[r_v] |= bit(r_u)
                    adj_mask[a_u] &= ~bit(a_v)
                    adj_mask[a_v] &= ~bit(a_u)
            
            if not swapped:
                # Try removing a random edge to escape local optimum
                if all_edges:
                    r_u, r_v = all_edges[0]
                    adj_mask[r_u] &= ~bit(r_v)
                    adj_mask[r_v] &= ~bit(r_u)
                    adj_mask = greedy_add(adj_mask)
        
        return adj_mask
>>>>>>> REPLACE


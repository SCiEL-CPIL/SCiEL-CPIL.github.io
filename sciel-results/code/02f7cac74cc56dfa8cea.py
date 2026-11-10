<<<<<<< SEARCH
        cur_count = edge_count(adj_masks)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj_masks[:]
            
            if best_count >= 58:
                if time.time() > deadline - 30:
                    break
=======
        cur_count = edge_count(adj_masks)
        
        if cur_count > best_count:
            best_count = cur_count
            best_adj = adj_masks[:]
            
            if best_count >= 58:
                if time.time() > deadline - 30:
                    break
>>>>>>> REPLACE


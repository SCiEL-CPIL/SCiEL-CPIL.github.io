    def local_search(block_indices):
        # ...
        local_start = time.time()
        local_deadline = local_start + 5.0 # Max 5s per local search? 
        # Or just check global deadline.
        
        # Inside loops:
        if time.time() > deadline - 5.0:
             return block_indices # Return current best

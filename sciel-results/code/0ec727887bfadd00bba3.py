    def has_independent_set_of_size_10(n, adj):
        """Check if there is an independent set of size 10."""
        
        def branch(remaining, size):
            if time.time() - start_time > DEADLINE - 2:
                return True # Assume valid if timeout? No, assume invalid to be safe? 
                             # Actually, if we timeout, we can't certify. 
                             # But for the search, if we can't prove it's < 10, we assume it's >= 10 and discard.
            if size == 10:
                return True
            if not remaining:
                return False
            if size + len(remaining) < 10:
                return False
            
            r = list(remaining)
            v = max(r, key=lambda x: sum(1 for y in r if y != x and y in adj[x]))
            
            # Include v
            if branch(set(r) - {v} - adj[v], size + 1):
                return True
            
            # Exclude v
            if branch(set(r) - {v}, size):
                return True
                
            return False

        return branch(set(range(n)), 0)

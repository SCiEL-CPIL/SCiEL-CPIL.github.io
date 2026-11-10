        def solve_with_backtracking(n, adj, lists):
            domains = [list(lists[i]) for i in range(n)]
            
            # Initial AC-3
            if not ac3(domains):
                return None
            
            node_count = 0
            NODE_LIMIT = 100000 # Should be plenty
            
            def search(domains):
                nonlocal node_count
                node_count += 1
                if node_count > NODE_LIMIT:
                    return None
                
                # Find variable with smallest domain
                min_len = float('inf')
                var = -1
                for i in range(n):
                    if len(domains[i]) < min_len:
                        min_len = len(domains[i])
                        var = i
                        if min_len == 1:
                            break
                
                if var == -1:
                    return None
                
                if min_len == 0:
                    return None
                
                # Check if solved
                if min_len == 1 and all(len(d) == 1 for d in domains):
                    return [d[0] for d in domains]
                
                # Try values
                for val in domains[var]:
                    saved_domains = [list(d) for d in domains]
                    domains[var] = [val]
                    
                    if ac3(domains):
                        result = search(domains)
                        if result:
                            return result
                    
                    # Restore
                    for i in range(n):
                        domains[i] = saved_domains[i]
                
                return None

            return search(domains)

        for i in range(n_jobs):
            job = current_perm[i]
            perm_without = current_perm[:i] + current_perm[i+1:]
            
            for j in range(n_jobs):
                if j == i:
                    continue
                # ...

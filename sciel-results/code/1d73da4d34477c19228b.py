<<<<<<< SEARCH
        # Local improvement
        selected = local_improve(selected, cliques)
        
        results.append(sorted(selected))
=======
        # Local improvement
        selected = local_improve(selected, cliques, set(range(n)))
        
        results.append(sorted(selected))
>>>>>>> REPLACE

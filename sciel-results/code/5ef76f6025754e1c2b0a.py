    if len(nodes) == 3:
        a, b, c = nodes
        # Determine the "best" pairing for the final bifurcation
        # Minimize sum of distances?
        # d[a,b] + d[c, ?] 
        # Just pick the smallest distance pair to join first.
        da_b = d[a, b]
        da_c = d[a, c]
        db_c = d[b, c]
        
        if da_b < da_c and da_b < db_c:
            return f"({trees[a]},{trees[b]},{trees[c]});" # Wait, this is still trifurcation in Newick syntax if not nested.
            # Newick ((a,b),c) is binary.
            # I should return f"(({trees[a]},{trees[b]}),{trees[c]});"
        elif da_c < da_b and da_c < db_c:
            return f"(({trees[a]},{trees[c]}),{trees[b]});"
        else:
            return f"(({trees[b]},{trees[c]}),{trees[a]});"

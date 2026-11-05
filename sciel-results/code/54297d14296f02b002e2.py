        if valid:
            # Add new products
            for p in new_suffixes:
                prods.add(p)
            
            dfs(idx + 1, current_seq + [x], prods, new_suffixes)
            
            # Undo
            for p in new_suffixes:
                prods.remove(p)

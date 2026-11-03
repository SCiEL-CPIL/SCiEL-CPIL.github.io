        current_girth = get_girth(adj_copy)
        if current_girth >= g:
            return adj_copy
        # ... try swaps ...
        # After swap:
        new_girth = get_girth(adj_copy)
        if new_girth >= current_girth:
            # Accept
            pass
        else:
            # Revert
            # ...
            continue

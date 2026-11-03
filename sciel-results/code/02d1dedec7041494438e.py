<<<<<<< SEARCH
    def full_local_search(code, deadline):
        """Combine multiple local search strategies."""
        code = local_search_remove_swap(code, deadline)
        code = local_search_critical(code, deadline)
        code = local_search_remove_swap(code, deadline)
        code = reduce_code(code)
        return code
=======
    def full_local_search(code, deadline):
        """Combine multiple local search strategies."""
        code = local_search_remove_swap(code, deadline)
        code = local_search_critical(code, deadline)
        code = local_search_remove_swap(code, deadline)
        code = reduce_code(code)
        # One more pass of remove_swap after reduction
        if time.time() < deadline:
            code = local_search_remove_swap(code, deadline)
        return code
>>>>>>> REPLACE


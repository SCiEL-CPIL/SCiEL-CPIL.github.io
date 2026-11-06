    def free_space(needed_set=None):
        if needed_set is None:
            needed_set = set()
        for x in list(fast):
            if x not in outputs_set and x not in needed_set:
                do_delete(x)
                return True
        # Fallback: delete any non-output
        for x in list(fast):
            if x not in outputs_set:
                do_delete(x)
                return True
        if fast:
            x = next(iter(fast))
            do_store(x)
            return True
        return False

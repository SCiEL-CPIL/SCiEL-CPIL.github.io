        # Ensure all outputs are stored
        for o in outputs:
            if o in fast_mem:
                ops.append(['store', o])
            else:
                # If not in fast mem, we must load it first? 
                # But if it's an output, it should have been computed/loaded.
                # If it was deleted (which shouldn't happen for outputs), we load then store.
                ops.append(['load', o])
                fast_mem.add(o)
                ops.append(['store', o])

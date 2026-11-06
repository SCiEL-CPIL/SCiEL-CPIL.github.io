        for o in outputs:
            if o not in fast_mem:
                ops.append(['load', o])
                fast_mem.add(o)
            ops.append(['store', o])
        
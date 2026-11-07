        v = min(remaining, key=lambda x: len([1 for pos, tf in vertex_to_triples[x] 
                                               if pos <= max_allowed and tf <= remaining]))

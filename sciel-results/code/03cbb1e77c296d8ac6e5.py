<<<<<<< SEARCH
    for o in outputs:
        if o not in fast:
            if o in avail:
                ops.append(["load", o])
                fast.add(o)
            else:
                ensure_blue(o)
        ops.append(["store", o])
=======
    for o in outputs:
        if o not in fast:
            # Ensure we have space to load o
            if len(fast) >= budget:
                # Evict a vertex that is not o (o is not in fast yet)
                # Prefer evicting a vertex that is not an output?
                # Simple heuristic: evict the one with smallest topo index
                victim = None
                min_idx = float('inf')
                for u in fast:
                    idx = topo_index.get(u, 0)
                    if idx < min_idx:
                        min_idx = idx
                        victim = u
                
                if victim is not None:
                    fast.discard(victim)
                    ops.append(["delete", victim])
            
            if o in avail:
                ops.append(["load", o])
                fast.add(o)
            else:
                ensure_blue(o)
        ops.append(["store", o])
>>>>>>> REPLACE

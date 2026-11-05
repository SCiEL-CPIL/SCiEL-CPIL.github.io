<<<<<<< SEARCH
        # Compute v
        if not computed[v]:
            operations.append(['compute', v])
            computed[v] = True
            in_fast[v] = True
            heapq.heappush(evict_heap, (last_needed[v], v))
        
        # After computing v, proactively evict vertices that are no longer needed
        # This helps keep memory usage low for future loads
        # We can evict any u in fast memory where last_needed[u] < v+1 and u is not an output
        # Actually, let's just let the make_room function handle evictions when needed
=======
        # Compute v
        if not computed[v]:
            operations.append(['compute', v])
            computed[v] = True
            in_fast[v] = True
            heapq.heappush(evict_heap, (last_needed[v], v))
        
        # After computing v, enforce memory limit by evicting if necessary
        # We call make_room with empty needed_vertices to evict any non-output vertex
        # that is not 'v' (though v is just added, and we might want to keep it)
        # Actually, we want to evict vertices to make room for future loads.
        # make_room will evict until sum(in_fast) < fast_memory_budget
        # It protects 'current_vertex' (we pass -1 so nothing protected except outputs)
        # and 'needed_vertices' (empty).
        make_room(set(), -1)
>>>>>>> REPLACE

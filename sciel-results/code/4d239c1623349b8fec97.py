"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def pebble(dag, fast_memory_budget):
    import sys
    sys.setrecursionlimit(10000)

    # Parse DAG
    # dag structure: assume it's a dict or similar with 'parents' and 'outputs' or similar.
    # Based on "dag has topologically ordered parents and output vertices", 
    # I will assume dag is a dict mapping vertex -> list of parents.
    # Or maybe dag is a list of (vertex, parents)?
    # The prompt says "dag has topologically ordered parents".
    # Let's assume standard format: dag is a dict {vertex: [parents]}
    # And we need to identify outputs. Outputs are vertices not in any other vertex's parent list.
    
    if not dag:
        return []

    # Identify all vertices
    vertices = list(dag.keys())
    
    # Build children map
    children = {v: [] for v in vertices}
    for v, parents in dag.items():
        for p in parents:
            children[p].append(v)

    # Identify outputs (vertices with no children)
    outputs = {v for v in vertices if not children[v]}
    
    # Topological sort (Kahn's algorithm or DFS)
    # Since parents are given, we can do a simple DFS or use the given order if available.
    # The prompt says "topologically ordered parents", which might imply the keys are in topological order?
    # "dag has topologically ordered parents" is ambiguous. It likely means the parents list is available.
    # Let's do a topological sort to be safe.
    
    in_degree = {v: 0 for v in vertices}
    for v, parents in dag.items():
        in_degree[v] = len(parents)
        
    # Wait, in_degree should be based on edges.
    # If dag[v] = [p1, p2], then p1->v and p2->v.
    # So out_degree of p1 increases.
    # For topological sort, we need nodes with 0 incoming edges (sources/inputs).
    # Sources are vertices with no parents.
    
    sources = [v for v in vertices if not dag[v]]
    
    # Queue for topological sort
    from collections import deque
    q = deque(sources)
    topo_order = []
    temp_in_degree = {v: len(dag[v]) for v in vertices}
    
    while q:
        u = q.popleft()
        topo_order.append(u)
        for w in children[u]:
            temp_in_degree[w] -= 1
            if temp_in_degree[w] == 0:
                q.append(w)
                
    if len(topo_order) != len(vertices):
        # Cycle detected or invalid DAG, fallback to simple order
        topo_order = vertices

    # Recursive solver
    def solve(sub_vertices, budget, keep_set):
        """
        sub_vertices: list of vertices to compute, in topological order relative to each other.
        budget: fast memory size.
        keep_set: set of vertices in sub_vertices that must be in fast memory at the end.
        """
        if not sub_vertices:
            return []

        # Base case: fits in memory
        if len(sub_vertices) <= budget:
            ops = []
            # Track external parents that are currently in fast memory
            loaded_external = set()
            
            # Map vertex to index in sub_vertices for quick lookup?
            # sub_vertices is a list, we can use set for O(1) lookup
            sub_set = set(sub_vertices)
            
            # Count how many children in sub_vertices each external parent has
            # to know when to delete
            ext_parent_child_count = {}
            for v in sub_vertices:
                for p in dag[v]:
                    if p not in sub_set:
                        if p not in ext_parent_child_count:
                            ext_parent_child_count[p] = 0
                        ext_parent_child_count[p] += 1

            for v in sub_vertices:
                # If v is in keep_set, it is assumed to be in memory already?
                # No, keep_set is what we MUST keep at the END.
                # At the START, only the keep_set from the PARENT call is in memory.
                # In this function, `keep_set` is the set to keep at the END.
                # The nodes in `keep_set` might not be in memory yet if they are not the first ones computed?
                # Actually, if v is in keep_set, we just compute it and don't delete/store it.
                
                # Ensure parents are in memory
                for p in dag[v]:
                    if p in sub_set:
                        # Parent is in this sub_problem, so it was computed earlier in this loop
                        pass
                    else:
                        # Parent is external
                        if p not in loaded_external:
                            ops.append(["load", p])
                            loaded_external.add(p)
                
                ops.append(["compute", v])
                
                # Delete external parents that are no longer needed
                for p in dag[v]:
                    if p not in sub_set:
                        ext_parent_child_count[p] -= 1
                        if ext_parent_child_count[p] == 0:
                            ops.append(["delete", p])
                            loaded_external.remove(p)

            # Finalize: store or delete nodes
            for v in sub_vertices:
                if v in keep_set:
                    # Keep in memory
                    continue
                else:
                    # Check if v is an output to the outside world
                    # i.e., has children not in sub_vertices
                    is_output_to_outside = False
                    for c in children[v]:
                        if c not in sub_set:
                            is_output_to_outside = True
                            break
                    
                    if is_output_to_outside:
                        ops.append(["store", v])
                    
                    ops.append(["delete", v])
            
            return ops

        # Recursive case: split
        mid = len(sub_vertices) // 2
        v1 = sub_vertices[:mid]
        v2 = sub_vertices[mid:]
        
        v1_set = set(v1)
        v2_set = set(v2)
        
        # Find interface: nodes in v1 that have children in v2
        interface = set()
        for v in v1:
            for c in children[v]:
                if c in v2_set:
                    interface.add(v)
                    break
        
        # Solve first half, keeping the interface in memory
        ops1 = solve(v1, budget, interface)
        
        # Solve second half, keeping the original keep_set
        # Note: The interface nodes are now in memory, which helps v2
        ops2 = solve(v2, budget, keep_set)
        
        return ops1 + ops2

    # Start with all vertices, keep_set is the set of global outputs?
    # No, the problem says "all outputs must finish blue".
    # "Blue" usually means in slow memory? Or in fast memory?
    # "Inputs start blue... all outputs must finish blue".
    # In pebbling, Blue = Slow Memory, Red = Fast Memory.
    # So outputs must be in Slow Memory at the end.
    # This means we should STORE all outputs.
    # So keep_set should be empty for the top-level call?
    # If keep_set is empty, we store all outputs.
    
    # However, if we keep some nodes in memory at the end, they are Red.
    # The requirement is "all outputs must finish blue".
    # So we must ensure all outputs are stored (Blue).
    # Thus, keep_set for the root call should be empty.
    
    return solve(topo_order, fast_memory_budget, set())

# EVOLVE-BLOCK-END

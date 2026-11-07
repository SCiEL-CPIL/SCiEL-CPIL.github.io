def construct():
    # The Fano plane is the unique 3-uniform hypergraph on 7 vertices
    # that is not 2-colorable. It has 7 edges.
    # We use vertices 0..6 for the Fano plane and leave 7,8 isolated.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # A valid 3-coloring for the Fano plane vertices 0..6
    # c(0)=0, c(1)=1, c(2)=2, c(3)=1, c(4)=2, c(5)=0, c(6)=1
    # Verification:
    # (0,1,2) -> 0,1,2 (OK)
    # (0,3,4) -> 0,1,2 (OK)
    # (0,5,6) -> 0,0,1 (OK)
    # (1,3,5) -> 1,1,0 (OK)
    # (1,4,6) -> 1,2,1 (OK)
    # (2,3,6) -> 2,1,1 (OK)
    # (2,4,5) -> 2,2,0 (OK)
    
    colors = [0, 1, 2, 1, 2, 0, 1, 0, 0]
    
    return {'edges': edges, 'colors': colors}

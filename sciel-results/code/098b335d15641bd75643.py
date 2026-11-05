"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Try Z_16 sum colouring
    # If valid, return it (16 colours)
    # Else, return seed (17 colours)
    
    def is_valid(colours):
        # Check all K4s
        # 16 vertices, 1820 K4s
        # 6 edges per K4
        # This is fast enough
        for i in range(16):
            for j in range(i+1, 16):
                for k in range(j+1, 16):
                    for l in range(k+1, 16):
                        # Get colours of edges (i,j), (i,k), (i,l), (j,k), (j,l), (k,l)
                        # Need a 2D array for colours
                        pass
        return True

    # Build Z_16 colouring
    cols = [[0]*16 for _ in range(16)]
    for i in range(16):
        for j in range(i+1, 16):
            c = (i + j) % 16
            cols[i][j] = c
            cols[j][i] = c
            
    # Check validity
    valid = True
    for i in range(16):
        for j in range(i+1, 16):
            for k in range(j+1, 16):
                for l in range(k+1, 16):
                    cs = {cols[i][j], cols[i][k], cols[i][l], cols[j][k], cols[j][l], cols[k][l]}
                    if len(cs) < 5:
                        valid = False
                        break
                if not valid: break
            if not valid: break
        if not valid: break
        
    if valid:
        return [cols[i][j] for i in range(16) for j in range(i+1, 16)]
    else:
        # Fallback to seed
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
# EVOLVE-BLOCK-END

<<<<<<< SEARCH
    U_final = U_4x4 + U_4x2
    V_final = []
    W_final = []
    
    # V_4x4 is 16 entries (4x4 B). We need to extend to 24 (4x6 B).
    for v in V_4x4:
        V_final.append(v + [0]*8) # Pad with 0s for cols 4,5
    
    V_final += V_4x2
    
    # W_4x4 is 16 entries (4x4 C). We need to extend to 24 (4x6 C).
    for w in W_4x4:
        W_final.append(w + [0]*8) # Pad with 0s for cols 4,5
        
    W_final += W_4x2
    
    return {"U": U_final, "V": V_final, "W": W_final}
=======
    U_final = U_4x4 + U_4x2
    V_final = []
    W_final = []
    
    # V_4x4 is 16 entries (4x4 B). We need to extend to 24 (4x6 B).
    for v in V_4x4:
        V_final.append(v + [0]*8) # Pad with 0s for cols 4,5
    
    V_final += V_4x2
    
    # W_4x4 is 16 entries (4x4 C). We need to extend to 24 (4x6 C).
    for w in W_4x4:
        W_final.append(w + [0]*8) # Pad with 0s for cols 4,5
        
    W_final += W_4x2
    
    # The interface expects shapes (n*m, rank), (m*p, rank), (p*n, rank)
    # Currently U_final is a list of 'rank' vectors of length n*m (16)
    # We need to transpose to get a list of n*m (16) vectors of length 'rank'
    
    rank = len(U_final)
    
    # Transpose U (16 x rank)
    U_out = [[U_final[r][i] for r in range(rank)] for i in range(16)]
    
    # Transpose V (24 x rank)
    V_out = [[V_final[r][i] for r in range(rank)] for i in range(24)]
    
    # Transpose W (24 x rank)
    W_out = [[W_final[r][i] for r in range(rank)] for i in range(24)]
    
    return {"U": U_out, "V": V_out, "W": W_out}
>>>>>>> REPLACE

   def construct():
       U44, V44, W44 = construct_4x4()
       U41, V41, W41 = construct_4x1()
       
       U = []
       V = []
       W = []
       
       # Add 44 part
       for i in range(49):
           U.append(U44[i])
           V.append(V44[i] + [0]*4) # Pad V to size 20
           W.append(W44[i] + [0]*4) # Pad W to size 20
           
       # Add 41 part
       for i in range(16):
           U.append(U41[i])
           V.append([0]*16 + V41[i]) # Pad V to size 20
           W.append([0]*16 + W41[i]) # Pad W to size 20
           
       return {"U": U, "V": V, "W": W}
   
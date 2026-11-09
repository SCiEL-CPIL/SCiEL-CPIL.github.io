# Check if N_v is an independent set
# Iterate pairs in N_v? Or just check if any node in N_v has a neighbor in N_v.
n_v_mask = N_v
while n_v_mask:
    lsb_w = n_v_mask & -n_v_mask
    w = lsb_w.bit_length() - 1
    n_v_mask ^= lsb_w
    if adj[w] & N_v: # Check if w is connected to any other node in N_v
         # Note: adj[w] & N_v includes w itself if self-loop, but simple graph.
         # It includes other nodes in N_v.
         # If non-zero, there is an edge within N_v.
         ok = False
         break

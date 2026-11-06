import numpy as np

def construct():
       p = 1009
       # Compute residues
       x = np.arange(1, p)
       residues = (x * x) % p
       
       # Boolean array for QR check
       is_qr = np.zeros(p, dtype=bool)
       is_qr[residues] = True
       
       # Map for roots (list is faster than dict for integer indexing in tight loops? Maybe.)
       root_map = [0] * p
       for i in range(1, p):
           r = (i * i) % p
           if root_map[r] == 0:
               root_map[r] = i
               
       # Search for minimal s
       # s must be such that s, s+1, s+2, s+3, s+4 are QRs
       # s+4 <= 1008 => s <= 1004
       for s in range(1, 1005):
           # Check slice
           if np.all(is_qr[s:s+5]):
               roots = [root_map[s+i] for i in range(5)]
               return {'start': s, 'roots': roots}
               
       raise RuntimeError('No valid run found')

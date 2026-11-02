   def evaluate(h):
       h = np.array(h)
       h = np.maximum(h, 0)
       l1 = np.sum(h)
       l2_sq = np.sum(h**2)
       conv = np.convolve(h, h, mode='full')
       max_conv = np.max(conv)
       if l1 == 0 or max_conv == 0: return 0
       return l2_sq / (l1 * max_conv)
   
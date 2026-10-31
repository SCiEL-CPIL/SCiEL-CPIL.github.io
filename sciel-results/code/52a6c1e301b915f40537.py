   r6 = r2 ** -3
   return 4.0 * (np.sum(r6 * r6 - r6) - np.trace(r6 * r6 - r6)) / 2.0
   
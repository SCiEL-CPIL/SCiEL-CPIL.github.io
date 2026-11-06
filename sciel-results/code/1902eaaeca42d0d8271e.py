      if k == 2:
          # 1/d1 + 1/d2 = num/den
          # d1*d2*num = d1*den + d2*den
          # d2*(d1*num - den) = d1*den
          # d2 = d1*den / (d1*num - den)
          for d1 in range(d_start, d_upper + 1):
              if time.time() > deadline: return None
              denom_d2 = d1 * num - den
              if denom_d2 <= 0: continue
              if (d1 * den) % denom_d2 != 0: continue
              d2 = (d1 * den) // denom_d2
              if d2 > d1 and d2 <= 10000000:
                  return path + [d1, d2]
          return None
      
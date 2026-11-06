  cur_len = len(divs)
  factor = 1
  for _ in range(e):
      factor *= p
      for j in range(cur_len):
          divs.append((divs[j][0] * factor, divs[j][1] | mask_bit))
  
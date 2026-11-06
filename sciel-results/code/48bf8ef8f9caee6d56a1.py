                old_cut = _cut_size(adj, labels)
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                new_cut = _cut_size(adj, labels)
